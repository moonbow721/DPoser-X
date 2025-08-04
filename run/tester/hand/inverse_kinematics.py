import random

from functools import partial

import os

import numpy as np
import torch
from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags
from torch.utils.data import DataLoader, random_split

from lib.algorithms.ik import DPoserIK
from lib.dataset.EvaSampler import DistributedEvalSampler
from lib.dataset.hand.evaler import IKEvaler
from lib.utils.generic import load_model
from lib.utils.misc import create_OpJtr_mask


try:
    from tensorboardX import SummaryWriter
except ImportError as e:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        print('Tensorboard is not Installed')

from lib.algorithms.advanced.model import create_model
import torch.distributed as dist
import torch.multiprocessing as mp
from lib.algorithms.advanced import sde_lib
from lib.dataset.hand import N_POSES, InterHandDataset
from lib.dataset.utils import Posenormalizer
from lib.body_model.hand_model import MANO


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='test diffusion model for handIK on ReInterHand dataset.')

    parser.add_argument('--ckpt-path', type=str, default='./pretrained_models/hand/BaseMLP/last.ckpt')
    parser.add_argument('--data_root', type=str, default='./data/hand_data',)
    parser.add_argument('--bodymodel-path', type=str, default='../body_models/mano',)

    parser.add_argument('--ik-type', type=str, default='sparse', choices=['only_end_visible', 'partial', 'sparse', 'noisy'])
    # optional
    parser.add_argument('--noise_std', type=float, default=0.002)   # 2mm
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100,)
    parser.add_argument('--gpus', type=int, help='num gpus to inference parallel')
    parser.add_argument('--port', type=str, help='master port of machines')

    args = parser.parse_args(argv[1:])

    return args


def get_dataloader(dataset, num_replicas=1, rank=0, batch_size=10000):
    sampler = DistributedEvalSampler(dataset,
                                     num_replicas=num_replicas,
                                     rank=rank,
                                     shuffle=False)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            sampler=sampler,
                            persistent_workers=False,
                            pin_memory=True,
                            drop_last=True)

    return dataloader


def setup(rank, world_size, port=None):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def inference(rank, args, config):
    print(f"Running DDP on rank {rank}.")
    setup(rank, args.gpus, args.port)
    batch_size = args.batch_size

    ## Load the pre-trained checkpoint from disk.
    device = torch.device("cuda", rank)
    POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
    model = create_model(config.model, N_POSES, POSE_DIM)
    model.to(device)
    model.eval()
    load_model(model, config.model, args.ckpt_path, device, is_ema=True)

    hand_model = MANO(model_path=args.bodymodel_path, batch_size=batch_size,).to(device)

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
    Normalizer = Posenormalizer(
        data_path=os.path.join(args.data_root, 'hand_normalizer'),
        normalize=config.data.normalize, min_max=config.data.min_max, rot_rep=config.data.rot_rep, device=device)
    normalizer_fn = partial(Normalizer.offline_normalize, from_axis=True)
    IK_fn = DPoserIK(model, hand_model, normalizer_fn, sde, config.training.continuous, )

    reinterhand_dataset = InterHandDataset(os.path.join(args.data_root, 'reinterhand_mocap.pt'), single_hand=True)
    # Spliting the whole reinterhand dataset
    torch.manual_seed(0)
    total_size = len(reinterhand_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    _, _, test_dataset = random_split(reinterhand_dataset, [train_size, val_size, test_size])

    test_loader = get_dataloader(test_dataset, num_replicas=args.gpus, rank=rank, batch_size=batch_size)

    if rank == 0:
        print(f'total samples: {len(test_dataset)}')

    all_results = []
    noise_to_weight = {0.002: 0.1, 0.005: 0.5, 0.01: 1.0}
    if args.ik_type == 'noisy':
        prior_weight = noise_to_weight.get(args.noise_std, 1.0)
        kwargs = {'t_max': 0.15, 't_min': 0.05, 'input_type': 'noisy', 'prior_weight': prior_weight}
    else:
        kwargs = {'t_max': 0.20, 't_min': 0.05, 'input_type': 'partial'}
    for _, batch_data in enumerate(test_loader):
        gts = batch_data['hand_pose'].to(device, non_blocking=True)
        hand_OpJtr = hand_model(hand_pose=gts).OpJtr
        if args.ik_type == 'noisy':
            mask_indices = None
            mask = torch.ones_like(hand_OpJtr)
            observation = hand_OpJtr + args.noise_std * torch.randn_like(hand_OpJtr)
        else:
            mask, observation, mask_indices, visible_indices = create_OpJtr_mask(hand_OpJtr, mask_type=args.ik_type, model='hand')
        pose_init = torch.randn(batch_size, N_POSES * POSE_DIM).to(device) * 0.01

        solution = IK_fn.optimize(observation, mask, pose_init, **kwargs)
        evaler = IKEvaler(hand_model=hand_model, mask_idx=mask_indices)
        eval_results = evaler.eval_hand(solution, gts)
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