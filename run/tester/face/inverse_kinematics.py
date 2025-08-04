import numpy as np
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from absl import app
from absl.flags import argparse_flags
from torch import nn

from lib.body_model.face_model import FLAME
from lib.dataset.EvaSampler import get_dataloader, setup, cleanup
from lib.dataset.face import FlameDataModule
from lib.dataset.face.evaler import IKEvaler
from lib.utils.generic import import_configs
from lib.utils.misc import create_joint_mask
from .face_prior import DPoser_pose_exp, DPoser_beta


class FaceIK(object):
    def __init__(self, prior_name, device, face_model, args):
        if prior_name == 'DPoser':
            DPoser_pose_exp_model = DPoser_pose_exp(device, args.exp_config_path, args)
            DPoser_beta_model = DPoser_beta(device, args.beta_config_path, args)
            self.prior_weights = {'pose_prior': 1.0, 'shape_prior': 1.0}
            self.prior_dict = {'shape': DPoser_beta_model, 'pose_expression': DPoser_pose_exp_model, }
        else:
            self.prior_weights = {}
            self.prior_dict = None

        self.face_model = face_model
        # L2 loss
        self.data_loss = nn.MSELoss(reduction='mean')

    def compute_prior_losses(self, shape, exp, jaw_pose, t):
        prior_losses = {}
        if 'shape' in self.prior_dict:
            prior_losses['shape_prior'] = self.prior_dict['shape'](shape, t) * self.prior_weights['shape_prior']
        if 'pose_expression' in self.prior_dict:
            input_data = torch.cat((jaw_pose, exp), dim=1)
            prior_losses['pose_expr_prior'] = self.prior_dict['pose_expression'](input_data, t) * self.prior_weights['pose_prior']
        if 'expression' in self.prior_dict:
            prior_losses['expr_prior'] = self.prior_dict['expression'](exp, t) * self.prior_weights['expr_prior']
        if 'pose' in self.prior_dict:
            prior_losses['pose_prior'] = self.prior_dict['pose'](jaw_pose, t) * self.prior_weights['pose_prior']
        return prior_losses

    def loss(self, shape, exp, jaw_pose, t):
        if self.prior_dict is None:
            return torch.tensor(0.0).to(t.device)
        else:
            prior_losses = self.compute_prior_losses(shape, exp, jaw_pose, t)
            return sum(prior_losses.values())

    def get_loss_weights(self, input_type='noisy'):
        """Set loss weights"""
        if input_type == 'partial':
            loss_weight = {'data': lambda cst, it: 100 * cst * (1 + it),
                           'prior': lambda cst, it: 0.001 * cst / (1 + it)}
        elif input_type == 'noisy':
            loss_weight = {'data': lambda cst, it: 500.0 * cst,
                           'prior': lambda cst, it: 0.001 * cst / (1 + it)}
        else:
            raise NotImplementedError
        return loss_weight

    @staticmethod
    def backward_step(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    def optimize(self, joints_3d, joints_mask, input_type='noisy',
                 t_max=0.15, t_min=0.05, t_fixed=0.1, time_strategy='3',
                 iterations=10, steps_per_iter=20, prior_weight=1.0,):
        batch_size = joints_3d.shape[0]
        joints_3d = joints_3d.detach()
        total_steps = iterations * steps_per_iter
        weight_dict = self.get_loss_weights(input_type)
        loss_dict = dict()
        with torch.enable_grad():
            shape = nn.Parameter(torch.zeros(batch_size, 100, device=joints_3d.device))
            exp = nn.Parameter(torch.zeros(batch_size, 100, device=joints_3d.device))
            jaw_pose = nn.Parameter(torch.zeros(batch_size, 3, device=joints_3d.device))
            params = [shape, exp, jaw_pose]
            learning_rates = [0.05, 0.05, 0.1]
            param_groups = [{'params': param, 'lr': lr} for param, lr in zip(params, learning_rates)]
            optimizer = torch.optim.Adam(param_groups)

            eps = 1e-3
            for it in range(iterations):
                for i in range(steps_per_iter):
                    step = it * steps_per_iter + i
                    optimizer.zero_grad()
                    '''   *************      Prior loss ***********         '''
                    if time_strategy == '1':
                        t = eps + torch.rand(1, device=joints_3d.device) * (1.0 - eps)
                    elif time_strategy == '2':
                        t = torch.tensor(t_fixed)
                    elif time_strategy == '3':
                        t = t_min + torch.tensor(total_steps - step - 1) / total_steps * (t_max - t_min)
                    else:
                        raise NotImplementedError
                    vec_t = torch.ones(batch_size, device=joints_3d.device) * t
                    loss_dict['prior'] = self.loss(shape, exp, jaw_pose, vec_t) * prior_weight
                    '''   ***********      Prior loss   ************       '''
                    opti_joints = self.face_model(betas=shape, expression=exp, jaw_pose=jaw_pose).Jtr
                    loss_dict['data'] = self.data_loss(opti_joints * joints_mask, joints_3d * joints_mask)

                    # Get total loss for backward pass
                    tot_loss = self.backward_step(loss_dict, weight_dict, it)
                    tot_loss.backward()
                    optimizer.step()

        return torch.cat([jaw_pose, exp, shape], dim=1).detach()



def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='test diffusion model for faceIK on WCPA dataset.')

    parser.add_argument('--data_root', type=str, default='./data/face_data',)
    parser.add_argument('--prior', type=str, default='DPoser', choices=['DPoser', 'None'],)
    parser.add_argument('--bodymodel-path', type=str, default='../body_models/flame',)

    parser.add_argument('--ik-type', type=str, default='noisy', choices=['left_face', 'right_face', 'half_face', 'noisy'])

    parser.add_argument('--exp-ckpt-path', type=str,
                        default='./pretrained_models/face/BaseMLP/last.ckpt')
    parser.add_argument('--beta-ckpt-path', type=str,
                        default='./pretrained_models/face_shape/BaseMLP/last.ckpt')
    parser.add_argument('--exp-config-path', type=str,
                        default='configs.face.subvp.pose_timefc.get_config')
    parser.add_argument('--beta-config-path', type=str,
                        default='configs.face.subvp.shape_timefc.get_config')

    # optional
    parser.add_argument('--noise_std', type=float, default=0.002)   # 2mm
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100,)
    parser.add_argument('--gpus', type=int, help='num gpus to inference parallel')
    parser.add_argument('--port', type=str, help='master port of machines')

    args = parser.parse_args(argv[1:])

    return args


def inference(rank, args, config):
    print(f"Running DDP on rank {rank}.")
    setup(rank, args.gpus, args.port)
    batch_size = args.batch_size

    ## Load the pre-trained checkpoint from disk.
    device = torch.device("cuda", rank)
    face_model = FLAME(model_path=args.bodymodel_path, batch_size=batch_size,
                       num_expressions=100, num_betas=100).to(device)

    IK_fn = FaceIK(args.prior, device, face_model, args)
    config.data.val_dataset_names = ['wcpapre_valid']
    data_module = FlameDataModule(config, args)
    data_module.setup(stage='test')
    test_dataset = data_module.test_dataset

    test_loader = get_dataloader(test_dataset, num_replicas=args.gpus, rank=rank, batch_size=batch_size)

    if rank == 0:
        print(f'total samples: {len(test_dataset)}')

    all_results = []
    noise_to_weight = {0.000: 0.1, 0.001: 0.5, 0.002: 1.0, 0.005: 2.0}
    if args.ik_type == 'noisy':
        prior_weight = noise_to_weight.get(args.noise_std, 1.0)
        kwargs = {'t_max': 0.20, 't_min': 0.05, 'prior_weight': prior_weight, 'input_type': 'noisy',}
    else:
        kwargs = {'t_max': 0.20, 't_min': 0.05, 'prior_weight': 0.1, 'input_type': 'partial',}
    for _, batch_data in enumerate(test_loader):
        gts = torch.cat([batch_data['jaw_pose'], batch_data['expression'], batch_data['betas']], dim=1).to(device)
        faceJtr = face_model(full_face_params=gts).Jtr
        if args.ik_type == 'noisy':
            mask_indices = None
            mask = torch.ones_like(faceJtr)
            observation = faceJtr + args.noise_std * torch.randn_like(faceJtr)
        else:
            mask, observation, mask_indices, visible_indices = create_joint_mask(faceJtr, mask_type=args.ik_type, model='face')
        solution = IK_fn.optimize(observation, mask, **kwargs)
        evaler = IKEvaler(face_model=face_model, mask_idx=mask_indices)
        eval_results = evaler.eval_face(solution, gts)
        all_results.append(eval_results)

    # collect data from other process
    print(f'rank[{rank}] subset len: {len(all_results)}')

    results_collection = [None for _ in range(args.gpus)]
    dist.gather_object(all_results, results_collection if rank == 0 else None, dst=0)

    if rank == 0:
        collected_results = np.concatenate(results_collection, axis=0)  # [batch_num,], every batch result is a dict
        collected_dict = {}

        # gather and settle results from all ranks
        for single_process_results in collected_results:
            for key, value in single_process_results.items():
                if key not in collected_dict:
                    collected_dict[key] = []
                collected_dict[key].extend(value)

        # compute the mean value
        for key, value in collected_dict.items():
            average_value = np.mean(np.array(value))
            print(f"The average of {key} is {average_value}")

    cleanup()


def main(args):
    # mp.freeze_support()
    mp.set_start_method('spawn')
    if args.port is None:
        args.port = random.randint(10000, 20000)
    config = import_configs(args.exp_config_path)

    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=inference, args=(rank, args, config))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)