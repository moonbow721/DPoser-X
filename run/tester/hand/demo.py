import os
from functools import partial
import random

import cv2
import numpy as np
import torch
from absl import flags, app
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags

from lib.algorithms.advanced import likelihood, sde_lib, sampling
from lib.algorithms.advanced.model import create_model
from lib.algorithms.ik import DPoserIK
from lib.body_model.hand_model import MANO
from lib.body_model.visual import render_mesh, multiple_render, vis_Ophand_skeletons
from lib.dataset.hand import N_POSES
from lib.dataset.hand.evaler import Evaler, IKEvaler
from lib.utils.generic import load_model
from lib.utils.misc import create_mask, create_OpJtr_mask
from lib.utils.metric import average_pairwise_distance, self_intersections_percentage, evaluate_fid, evaluate_prdc, \
    evaluate_dnn

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Visualizing configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

from lib.dataset.utils import Posenormalizer

bg_img = np.ones([192*2, 256*2, 3]) * 255  # background canvas
focal = [5000*2, 5000*2]
princpt = [80*2, 128*2]


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='DPoser-hand demo on toy data')

    parser.add_argument('--file-path', type=str, default='./examples/toy_hand_data.npz', help='saved npz file')
    parser.add_argument('--task', type=str, default='view', choices=['view',
                                                                     'generation',
                                                                     'eval_generation',
                                                                     'inverse_kinematics',
                                                                     ])
    parser.add_argument('--mode', default='DPoser', choices=['DPoser', 'ScoreSDE', 'MCG', 'DPS'],
                        help='different solvers for completion task')
    parser.add_argument('--hypo', type=int, default=10, help='multi hypothesis prediction for completion')
    parser.add_argument('--part', type=str, default='middle_finger',
                        help='the masked part for completion task')

    parser.add_argument('--ik-type', type=str, default='sparse',
                        choices=['only_end_visible', 'partial', 'sparse', 'noisy'])
    parser.add_argument('--noise_std', type=float, default=0.005)  # 2mm

    parser.add_argument('--bodymodel-path', type=str, default='../body_models/mano')
    parser.add_argument('--data-path', type=str, default='./data/hand_data')
    parser.add_argument('--ckpt-path', type=str, default='./pretrained_models/hand/BaseMLP/last.ckpt',
                        help='load trained diffusion model')

    parser.add_argument('--view', type=str, default='half_right_bottom', help='render direction')
    parser.add_argument('--faster', action='store_true', help='faster render (lower quality)')
    parser.add_argument('--output-path', type=str, default='./output/hand/test_results')
    parser.add_argument('--device', type=str, default='cuda:4')

    args = parser.parse_args(argv[1:])

    return args


def main(args):
    torch.manual_seed(42)
    random.seed(42)
    """
    *****************        load some gt samples and view       *****************
    """
    save_renders = partial(multiple_render, part='hand', bg_img=bg_img, focal=focal, princpt=princpt,
                           device=args.device)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    sample_num = 50
    body_model = MANO(model_path=args.bodymodel_path, batch_size=sample_num).to(args.device)

    """
    *****************        model preparation for demo tasks       *****************    
    """
    config = FLAGS.config
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
    model = create_model(config.model, N_POSES, POSE_DIM)
    model.to(args.device)
    model.eval()
    load_model(model, config.model, args.ckpt_path, args.device, is_ema=True)

    inverse_scaler = lambda x: x
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, rtol=1e-4, atol=1e-4, eps=1e-4)

    Normalizer = Posenormalizer(
        data_path=os.path.join(args.data_path, 'hand_normalizer'),
        normalize=config.data.normalize, min_max=config.data.min_max, rot_rep=config.data.rot_rep, device=args.device)
    normalizer_fn = Normalizer.offline_normalize
    denormalize_fn = Normalizer.offline_denormalize

    if args.task == 'generation':
        target_path = os.path.join(args.output_path, 'generation')

        sample_num = 50
        sampling_shape = (sample_num, N_POSES * POSE_DIM)
        default_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                                   device=args.device)
        _, samples = default_sampler(model, observation=None)
        body_model = MANO(model_path=args.bodymodel_path, batch_size=sample_num).to(args.device)

        save_renders(samples, denormalize_fn, body_model, target_path, 'generated_sample{}.png',
                     faster=args.faster, view=args.view)
        print(f'samples saved under {target_path}')

    elif args.task == 'eval_generation':
        sample_num = 50000
        sampling_shape = (sample_num, N_POSES * POSE_DIM)
        sampling_eps = 5e-3
        sampling.method = 'pc'  # pc or ode
        default_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                                   device=args.device)
        _, samples = default_sampler(model, observation=None)
        samples = denormalize_fn(samples, to_axis=True)
        fid = evaluate_fid(samples, f'{args.data_path}/statistics.npz')
        print('FID for 50000 generated samples', fid)
        prdc = evaluate_prdc(samples, f'{args.data_path}/reference_batch.pt')
        print(prdc)
        if os.path.exists(f'{args.data_path}/all_data.pt'):
            dnn = evaluate_dnn(samples, f'{args.data_path}/all_data.pt',)
            print('DNN for 50000 generated samples', dnn)
        samples = samples[:500]
        body_model = MANO(model_path=args.bodymodel_path, batch_size=500).to(args.device)
        body_out = body_model(hand_pose=samples)
        joints3d = body_out.OpJtr
        APD = average_pairwise_distance(joints3d)
        print('average_pairwise_distance for 500 generated samples', APD)

        return

    """
    *****************        data preparation for demo tasks       *****************    
    """
    data = np.load(args.file_path, allow_pickle=True)
    sample_num = 200
    body_poses = data['pose_samples'][:sample_num]
    print(f'loaded axis pose data {body_poses.shape} from {args.file_path}')
    body_poses = torch.from_numpy(body_poses).to(args.device)
    body_model = MANO(model_path=args.bodymodel_path, batch_size=sample_num).to(args.device)

    if args.task == 'view':
        target_path = os.path.join(args.output_path, 'view')
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        save_renders(body_poses, None, body_model, target_path, 'GT_sample{}.png',
                     convert=False, faster=args.faster, view=args.view)
        print(f'rendered images saved under {target_path}')
        return

    elif args.task == 'inverse_kinematics':
        normalizer_fn = partial(Normalizer.offline_normalize, from_axis=True)
        IK_fn = DPoserIK(model, body_model, normalizer_fn, sde, config.training.continuous, )

        gts = body_poses
        hand_OpJtr = body_model(hand_pose=gts).OpJtr
        if args.ik_type == 'noisy':
            mask_indices = None
            mask = torch.ones_like(hand_OpJtr)
            observation = hand_OpJtr + args.noise_std * torch.randn_like(hand_OpJtr)
            markers = observation.detach().cpu().numpy()
        elif args.ik_type == 'partial' or args.ik_type == 'sparse':
            # process each sample one by one and concat to batch
            for s_idx in range(sample_num):
                s_mask, s_observation, s_mask_indices, s_visible_indices = create_OpJtr_mask(hand_OpJtr[s_idx:s_idx+1], mask_type=args.ik_type, model='hand')
                if s_idx == 0:
                    mask, observation, mask_indices, visible_indices = s_mask, s_observation, s_mask_indices, s_visible_indices
                    markers = hand_OpJtr[s_idx].detach().cpu().numpy()[s_visible_indices]
                else:
                    mask = torch.cat((mask, s_mask), dim=0)
                    observation = torch.cat((observation, s_observation), dim=0)
                    mask_indices = np.concatenate((mask_indices, s_mask_indices), axis=0)
                    markers = np.concatenate((markers, hand_OpJtr[s_idx].detach().cpu().numpy()[s_visible_indices]), axis=0)
            markers = markers.reshape(sample_num, -1, 3)
        else:
            mask, observation, mask_indices, visible_indices = create_OpJtr_mask(hand_OpJtr, mask_type=args.ik_type, model='hand')
            markers = hand_OpJtr.detach().cpu().numpy()[:, visible_indices]

        pose_init = torch.randn(sample_num, N_POSES * POSE_DIM).to(args.device) * 0.01
        input_type = 'noisy' if args.ik_type == 'noisy' else 'partial'
        solution = IK_fn.optimize(observation, mask, pose_init, input_type=input_type)

        evaler = IKEvaler(hand_model=body_model, mask_idx=mask_indices)
        eval_results = evaler.eval_hand(solution, gts)
        evaler.print_eval_result(eval_results,)

        target_path = os.path.join(args.output_path, 'inverse_kinematics', args.ik_type)
        save_renders(gts, denormalize_fn, body_model, target_path, 'sample{}_original.png', markers,
                     convert=False, faster=args.faster, view=args.view)
        print(f'Original samples with markers under {target_path}')

        save_renders(solution, denormalize_fn, body_model, target_path, convert=False,
                     img_name='sample{}_solution.png', faster=args.faster, view=args.view)
        print(f'Solution samples under {target_path}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    app.run(main, flags_parser=parse_args)