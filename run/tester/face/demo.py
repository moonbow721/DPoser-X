import os
from functools import partial

import cv2
import numpy as np
import torch
from absl import flags, app
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags

from lib.algorithms.advanced import likelihood, sde_lib, sampling
from lib.algorithms.advanced.model_fullface import create_fullface_model
from lib.body_model.face_model import FLAME
from lib.body_model.visual import render_mesh, multiple_render
from lib.dataset.face.evaler import IKEvaler
from lib.utils.misc import slerp, create_joint_mask
from lib.utils.metric import evaluate_fid, evaluate_prdc, evaluate_dnn
from .inverse_kinematics import FaceIK
from lib.dataset.utils import CombinedNormalizer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Visualizing configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])


bg_img = np.ones([256, 256, 3]) * 255  # background canvas
focal = [5000, 5000]
princpt = [128, 115]


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='DPoser-face demo on toy data')

    parser.add_argument('--file-path', type=str, default='./examples/toy_face_data.npz', help='saved npz file')
    parser.add_argument('--task', type=str, default='view', choices=['view',
                                                                     'generation',
                                                                     'eval_generation',
                                                                     'inverse_kinematics',
                                                                     ])
    parser.add_argument('--bodymodel-path', type=str, default='../body_models/flame')
    parser.add_argument('--data-root', type=str, default='./data/face_data', help='normalizer folder')

    # For IK solver
    parser.add_argument('--exp-ckpt-path', type=str,
                        default='./pretrained_models/face/BaseMLP/last.ckpt')
    parser.add_argument('--beta-ckpt-path', type=str,
                        default='./pretrained_models/face_shape/BaseMLP/last.ckpt')
    parser.add_argument('--exp-config-path', type=str,
                        default='configs.face.subvp.pose_timefc.get_config')
    parser.add_argument('--beta-config-path', type=str,
                        default='configs.face.subvp.shape_timefc.get_config')
    parser.add_argument('--ik-type', type=str, default='noisy',
                        choices=['left_face', 'right_face', 'half_face', 'noisy'])
    parser.add_argument('--noise_std', type=float, default=0.002)  # 2mm

    parser.add_argument('--view', type=str, default='front', help='render direction')
    parser.add_argument('--faster', action='store_true', help='faster render (lower quality)')
    parser.add_argument('--output-path', type=str, default='./output/face_full/test_results')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args(argv[1:])

    return args


def main(args):
    torch.manual_seed(42)
    """
    *****************        load some gt samples and view       *****************
    """
    save_renders = partial(multiple_render, part='full-face', bg_img=bg_img, focal=focal, princpt=princpt,
                           device=args.device)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

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
    POSES_LIST = [103, 100]
    if config.data.rot_rep == 'rot6d':
        POSES_LIST[0] = 106
    N_POSES = POSES_LIST[0] + POSES_LIST[1]
    model = create_fullface_model(config.model, POSES_LIST, POSE_DIM=1)
    model.to(args.device)
    model.eval()

    inverse_scaler = lambda x: x
    data_path_dict = {"jaw_pose": os.path.join(args.data_root, 'jaw_normalizer'),
                      "expression": os.path.join(args.data_root, 'expression_normalizer'),
                      "betas": os.path.join(args.data_root, 'betas_normalizer')}
    Normalizer = CombinedNormalizer(
        data_path_dict=data_path_dict, model='full-face',
        normalize=config.data.normalize, min_max=config.data.min_max, rot_rep=config.data.rot_rep, device=args.device)
    denormalize_fn = Normalizer.offline_denormalize

    if args.task == 'generation':
        target_path = os.path.join(args.output_path, 'generation')

        sample_num = 100
        sampling_shape = (sample_num, N_POSES)
        default_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                                   device=args.device)
        _, samples = default_sampler(model, observation=None)
        body_model = FLAME(model_path=args.bodymodel_path, num_betas=100, batch_size=sample_num).to(args.device)
        samples = denormalize_fn(samples, to_axis=True)
        expression, betas = samples[:, :103], samples[:, 103:]
        # The first 50 sample, same expression, different shape
        expression[:50] = torch.zeros_like(expression[:50])
        # The second 50 sample, same shape, different expression
        betas[50:] = betas[0].repeat(50, 1)

        save_renders(samples, None, body_model, target_path, 'generated_sample{}.png', convert=False,
                     faster=args.faster, view=args.view)
        print(f'samples saved under {target_path}')
        return
    elif args.task == 'eval_generation':
        '''
        evaluate FID, PRDC, and DNN
        '''

        sample_num = 50000
        sampling_shape = (sample_num, N_POSES)
        # sampling_eps = 5e-3
        default_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                                   device=args.device)
        _, samples = default_sampler(model, observation=None)
        samples = denormalize_fn(samples, to_axis=True)
        expression, betas = samples[:, :103], samples[:, 103:]
        data_root = '/data3/ljz24/projects/3d/DPoser/face_data'
        for mode, data in zip(['expression', 'shape'], [expression, betas]):
            print(f'evaluating {mode}')
            fid = evaluate_fid(data, f'{data_root}/statistics_{mode}.npz')
            print('FID for 50000 generated samples', fid)
            prdc = evaluate_prdc(data, f'{data_root}/reference_batch_{mode}.pt')
            print(prdc)
            if os.path.exists(f'{data_root}/all_data_{mode}.pt'):
                dnn = evaluate_dnn(data, f'{data_root}/all_data_{mode}.pt', measure='absolute', batch_size=50000)
                print('DNN for 50000 generated samples', dnn)

        return
    
    """
    *****************        data preparation for demo tasks       *****************    
    """
    data = np.load(args.file_path, allow_pickle=True)
    print(list(data.keys()))
    sample_num = 50
    body_poses = np.concatenate([data['jaw_pose'], data['expression'], data['betas'][:, :100]], axis=1)[:sample_num]
    body_model = FLAME(model_path=args.bodymodel_path, num_betas=100, batch_size=sample_num).to(args.device)
    print(f'loaded axis pose data {body_poses.shape} from {args.file_path}')
    body_poses = torch.from_numpy(body_poses).to(args.device).float()

    if args.task == 'view':
        target_path = os.path.join(args.output_path, 'view')
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        save_renders(body_poses, None, body_model, target_path, 'GT_sample{}.png',
                     convert=False, faster=args.faster, view=args.view)
        print(f'rendered images saved under {target_path}')
        return
    elif args.task == 'inverse_kinematics':
        IK_fn = FaceIK('DPoser', args.device, body_model, args)
        gts = body_poses
        faceJtr = body_model(full_face_params=gts).Jtr
        if args.ik_type == 'noisy':
            mask_indices = None
            mask = torch.ones_like(faceJtr)
            observation = faceJtr + args.noise_std * torch.randn_like(faceJtr)
            markers = observation.detach().cpu().numpy()
            noise_to_weight = {0.000: 0.1, 0.001: 0.1, 0.002: 1.0, 0.003: 1.0, 0.005: 2.0}
            prior_weight = noise_to_weight.get(args.noise_std, 1.0)
            input_type = 'noisy'
        else:
            mask, observation, mask_indices, visible_indices = create_joint_mask(faceJtr, mask_type=args.ik_type,
                                                                                 model='face')
            markers = faceJtr.detach().cpu().numpy()[:, visible_indices]
            prior_weight = 1.0
            input_type = 'partial'

        solution = IK_fn.optimize(observation, mask, prior_weight=prior_weight, input_type=input_type,)
        evaler = IKEvaler(face_model=body_model, mask_idx=mask_indices)
        eval_results = evaler.eval_face(solution, gts)
        evaler.print_eval_result(eval_results, )

        ik_name = args.ik_type if args.ik_type != 'noisy' else f'noisy_{round(args.noise_std*1000)}'
        target_path = os.path.join(args.output_path, 'inverse_kinematics', ik_name,)
        save_renders(gts, denormalize_fn, body_model, target_path, 'sample{}_original.png', markers=markers,
                     convert=False, faster=args.faster, view=args.view)
        print(f'Original samples with markers under {target_path}')

        save_renders(solution, denormalize_fn, body_model, target_path, convert=False,
                     img_name='sample{}_solution.png', faster=args.faster, view=args.view)
        print(f'Solution samples under {target_path}')

        save_renders(gts, denormalize_fn, body_model, target_path, 'sample{}_gt.png',
                     convert=False, faster=args.faster, view=args.view)
        print(f'Original samples without markers under {target_path}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    app.run(main, flags_parser=parse_args)