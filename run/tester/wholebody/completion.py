import os
from types import SimpleNamespace

import numpy as np
import torch
from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags
import torch.distributed as dist
import torch.multiprocessing as mp

from lib.algorithms.advanced.model_wholebody import create_wholebody_model
from lib.algorithms.completion import DPoserComp
from lib.algorithms.advanced import sde_lib, sampling
from lib.dataset.EvaSampler import DistributedEvalSampler, get_dataloader
from lib.dataset.whole_body import SmplxDataModule, POSES_LIST, EMAGEDataset, WholebodyWrapper, \
    ArcticDataset
from lib.dataset.whole_body.evaler import Evaler
from lib.dataset.utils import CombinedNormalizer
from lib.utils.generic import load_pl_weights, load_model
from lib.utils.misc import create_wholebody_mask
from lib.body_model.body_model import BodyModel
from lib.utils.metric import average_pairwise_distance, self_intersections_percentage

try:
    from tensorboardX import SummaryWriter
except ImportError as e:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        print('Tensorboard is not Installed')

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='test diffusion model for completion')

    parser.add_argument('--dataset', default='egobody', choices=['egobody', 'arctic', 'emage'])
    parser.add_argument('--ckpt-path', type=str, default='./pretrained_models/wholebody/mixed/last.ckpt')
    parser.add_argument('--data-path', type=str, default='./data', help='normalizer folder')
    parser.add_argument('--data-root', type=str, default='./data/wholebody_data', help='dataset root')
    parser.add_argument('--bodymodel-path', type=str, default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                        help='path of SMPLX model')

    parser.add_argument('--hypo', type=int, default=10, help='number of hypotheses to sample')
    parser.add_argument('--part', type=str, default='one_hand', choices=['lhand', 'rhand', 'one_hand', 'face'])
    # optional
    parser.add_argument('--mode', default='DPoser', choices=['DPoser', 'ScoreSDE', 'MCG', 'DPS'])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--sample', type=int, help='sample testset to reduce data for other tasks')
    parser.add_argument('--batch_size', type=int, default=100, )
    parser.add_argument('--gpus', type=int, help='num gpus to inference parallel')
    parser.add_argument('--port', type=str, default='14600', help='master port of machines')

    args = parser.parse_args(argv[1:])

    return args


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def inference(rank, args, config):
    print(f"Running DDP on rank {rank}.")
    setup(rank, args.gpus, args.port)

    ## Load the pre-trained checkpoint from disk.
    device = torch.device("cuda", rank)
    POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
    DATA_DIM = POSES_LIST[0] * POSE_DIM + 2 * POSES_LIST[1] * POSE_DIM + POSES_LIST[2]  # two hands
    model = create_wholebody_model(config.model, POSES_LIST, POSE_DIM)
    if config.model.type != 'Combiner':
        load_model(model, config.model, args.ckpt_path, device, is_ema=True)
    model.to(device)
    model.eval()

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=args.steps)
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=args.steps)
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=args.steps)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Setup sampling functions
    comp_fn = DPoserComp(model, sde, config.training.continuous, )
    data_path = {"body_pose": os.path.join(args.data_path, 'body_data', 'body_normalizer'),
                 "left_hand_pose": os.path.join(args.data_path, 'hand_data', 'hand_normalizer'),
                 "right_hand_pose": os.path.join(args.data_path, 'hand_data', 'hand_normalizer'),
                 "jaw_pose": os.path.join(args.data_path, 'face_data', 'jaw_normalizer'),
                 "expression": os.path.join(args.data_path, 'face_data', 'expression_normalizer')}
    Normalizer = CombinedNormalizer(
        data_path_dict=data_path, model='whole-body',
        normalize=config.data.normalize, min_max=config.data.min_max, rot_rep=config.data.rot_rep, device=device)

    # Perform completion baselines (ScoreSDE, MCG, DPS)
    task_args = SimpleNamespace(task=None)
    if args.mode == 'DPS':
        task_args.task, inverse_solver = 'default', 'BP'
    elif args.mode == 'MCG':
        task_args.task, inverse_solver = 'completion', 'BP'
    elif args.mode == 'ScoreSDE':
        task_args.task, inverse_solver = 'completion', None
    else:  # plain generation sampler
        task_args.task, inverse_solver = 'default', None
    comp_sampler = sampling.get_sampling_fn(config, sde, (args.batch_size, DATA_DIM),
                                            lambda x: x, 1e-3, device=device, inverse_solver=inverse_solver)

    batch_size = args.batch_size
    if args.dataset == 'egobody':
        data_module = SmplxDataModule(config, args)
        data_module.setup(stage='test')
        test_dataset = data_module.test_dataset
    elif args.dataset == 'arctic':
        test_dataset = WholebodyWrapper(
            ArcticDataset(dataset_root=os.path.join(args.data_root, 'Arctic', 'merged_smplx'),
                          split='val', sample_interval=args.sample))
    elif args.dataset == 'emage':
        test_dataset = WholebodyWrapper(EMAGEDataset(dataset_root=os.path.join(args.data_root, 'BEAT'),
                                                     split='test', sample_interval=args.sample))
    else:
        raise NotImplementedError(f"Dataset {args.dataset} unknown.")
    test_loader = get_dataloader(test_dataset, num_replicas=args.gpus, rank=rank, batch_size=batch_size)

    body_model = BodyModel(bm_path=args.bodymodel_path,
                           num_betas=10,
                           batch_size=batch_size,
                           model_type='smplx').to(device)

    if rank == 0:
        print(f'total samples with reduction: {len(test_dataset)}')

    all_results = []

    for _, batch in enumerate(test_loader):
        gts = torch.cat([batch['body_pose'], batch['left_hand_pose'], batch['right_hand_pose'],
                           batch['jaw_pose'], batch['expression']], dim=1).to(device, non_blocking=True)
        poses = Normalizer.offline_normalize(gts, from_axis=True)
        mask, observation, mask_type = create_wholebody_mask(poses, part=args.part, mask_root=True)
        evaler = Evaler(body_model=body_model, part=mask_type)

        multihypo_denoise = []
        for hypo in range(args.hypo):
            if args.mode == 'DPoser':
                completion = comp_fn.optimize(observation, mask, lr=0.1)
            else:
                _, completion = comp_sampler(model, observation=observation, mask=mask, args=task_args)
            multihypo_denoise.append(completion)
        multihypo_denoise = torch.stack(multihypo_denoise, dim=1)

        preds = Normalizer.offline_denormalize(multihypo_denoise, to_axis=True)
        # *************** Compute APD ********************   preds: [b, hypo, j*3]
        apd_body_model = BodyModel(bm_path='/data3/ljz24/projects/3d/body_models/smplx',
                                   model_type='smplx',
                                   batch_size=args.hypo,
                                   num_betas=10).to(device)
        batch_APD = []
        joint_idx = evaler.joint_idx
        with torch.no_grad():
            for b in range(batch_size):
                samples = preds[b]
                body_out = apd_body_model(wholebody_params=samples)
                joints3d = body_out.Jtr
                body_joints3d = joints3d[:, joint_idx, :]
                APD = average_pairwise_distance(body_joints3d) * 100.0  # m -> cm
                batch_APD.append(APD)
        all_results.append({'APD': batch_APD})
        # *************** Compute APD ********************

        eval_results = evaler.multi_eval_bodys_all(preds, gts)  # [batch_size, ]
        all_results.append(eval_results)

    # collect data from other process
    print(f'rank[{rank}] subset len: {len(all_results)}')

    results_collection = [None for _ in range(args.gpus)]
    dist.gather_object(
        all_results,
        results_collection if rank == 0 else None,
        dst=0
    )

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
    print(f"Testing on {args.dataset} dataset with {args.mode} model, {args.hypo} hypotheses, mask part {args.part}")
    mp.set_start_method('spawn')

    config = FLAGS.config

    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=inference, args=(rank, args, config))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)