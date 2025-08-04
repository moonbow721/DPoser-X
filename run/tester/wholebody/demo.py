import os
from functools import partial
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from absl import flags, app
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags

from lib.algorithms.advanced import likelihood, sde_lib, sampling
from lib.algorithms.advanced.model_wholebody import create_wholebody_model
from lib.algorithms.completion import DPoserComp
from lib.body_model.body_model import BodyModel
from lib.body_model.hand_model import MANO
from lib.body_model.visual import render_mesh, multiple_render
from lib.dataset.whole_body.evaler import Evaler
from lib.utils.generic import load_model
from lib.utils.metric import evaluate_fid, evaluate_prdc, average_pairwise_distance, \
    self_intersections_percentage, evaluate_dnn, average_pairwise_distance_wholebody
from lib.utils.misc import slerp, create_wholebody_mask
from lib.dataset.utils import CombinedNormalizer, Posenormalizer


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Visualizing configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])


bg_img = np.ones([512 * 4, 384 * 4, 3]) * 255  # background canvas
focal = [1500 * 4, 1500 * 4]
princpt = [200 * 4, 192 * 4]


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='DPoser-X demo on toy data')

    parser.add_argument('--file-path', type=str, default='./examples/toy_wholebody_data.npz', help='saved npz file')
    parser.add_argument('--task', type=str, default='view', choices=['view',
                                                                     'generation',
                                                                     'eval_generation',
                                                                     'completion'
                                                                     ])

    parser.add_argument('--bodymodel-path', type=str, default='../body_models/smplx')
    parser.add_argument('--data-path', type=str, default='./data', help='normalizer folder')
    parser.add_argument('--ckpt-path', type=str,
                        default='./pretrained_models/wholebody/mixed/last.ckpt',
                        help='load trained diffusion model')

    parser.add_argument('--hypo', type=int, default=10, help='multi hypothesis prediction for completion')
    parser.add_argument('--part', type=str, default='lhand', choices=['lhand', 'rhand', 'face'],
                        help='the masked part for completion task')
    parser.add_argument('--mode', default='DPoser', choices=['DPoser', 'ScoreSDE', 'MCG', 'DPS', 'DSG'],
                        help='different solvers for completion task')

    parser.add_argument('--view', type=str, default='front', help='render direction')
    parser.add_argument('--faster', action='store_true', help='faster render (lower quality)')
    parser.add_argument('--output-path', type=str, default='./output/wholebody/test_results')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args(argv[1:])

    return args


def main(args):
    torch.manual_seed(0)
    """
    *****************        load some gt samples and view       *****************
    """
    save_renders = partial(multiple_render, part='whole-body', bg_img=bg_img, focal=focal, princpt=princpt,
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
    POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
    POSES_LIST = [21, 15, 100 + 3]
    if config.data.rot_rep == 'rot6d':
        POSES_LIST[2] = 100 + 6
    DATA_DIM = POSES_LIST[0] * POSE_DIM + 2 * POSES_LIST[1] * POSE_DIM + POSES_LIST[2]  # two hands
    model = create_wholebody_model(config.model, POSES_LIST, POSE_DIM)
    if config.model.type != 'Combiner':
        load_model(model, config.model, args.ckpt_path, args.device, is_ema=True)
    model.to(args.device)
    model.eval()

    inverse_scaler = lambda x: x
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, rtol=1e-4, atol=1e-4, eps=1e-4)

    # We flip the left hand pose input before normalizing and flip back after denormalizing in CombinedNormalizer
    data_path = {"body_pose": os.path.join(args.data_path, 'body_data', 'body_normalizer'),
                 "left_hand_pose": os.path.join(args.data_path, 'hand_data', 'hand_normalizer'),
                 "right_hand_pose": os.path.join(args.data_path, 'hand_data', 'hand_normalizer'),
                 "jaw_pose": os.path.join(args.data_path, 'face_data', 'jaw_normalizer'),
                 "expression": os.path.join(args.data_path, 'face_data', 'expression_normalizer')}
    Normalizer = CombinedNormalizer(
        data_path_dict=data_path, model='whole-body',
        normalize=config.data.normalize, min_max=config.data.min_max, rot_rep=config.data.rot_rep, device=args.device)
    normalizer_fn = Normalizer.offline_normalize
    denormalize_fn = Normalizer.offline_denormalize

    if args.task == 'generation':
        target_path = os.path.join(args.output_path, 'generation',)
        sample_num = 50
        sampling_shape = (sample_num, DATA_DIM)
        default_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                                   device=args.device)
        _, samples = default_sampler(model, observation=None)
        body_model = BodyModel(bm_path=args.bodymodel_path, num_betas=10, model_type='smplx',
                               batch_size=sample_num).to(args.device)

        save_renders(samples, denormalize_fn, body_model, target_path, 'generated_sample{}.png',
                     faster=args.faster, view=args.view)
        print(f'samples saved under {target_path}')

        from lib.body_model.face_model import FLAME
        data_path = {"jaw_pose": os.path.join(args.data_path, 'face_data', 'jaw_normalizer'),
                     "expression": os.path.join(args.data_path, 'face_data', 'expression_normalizer')}
        Normalizer = CombinedNormalizer(
            data_path_dict=data_path, model='face',
            normalize=config.data.normalize, min_max=config.data.min_max, rot_rep=config.data.rot_rep,
            device=args.device)
        denormalize_fn = Normalizer.offline_denormalize
        save_renders_local = partial(multiple_render, part='face', bg_img=np.ones([256, 256, 3]) * 255,
                                     focal=[5000, 5000],
                                     princpt=[128, 115], device=args.device)
        face_samples = samples[:, 63 + 90:]
        body_model = FLAME(model_path='../body_models/flame', batch_size=sample_num).to(args.device)
        save_renders_local(face_samples, denormalize_fn, body_model, target_path, 'generated_face_sample{}.png',
                           faster=args.faster, view=args.view)
        print(f'face samples saved under {target_path}')

        Normalizer = Posenormalizer(
            data_path=os.path.join(args.data_path, 'hand_data', 'hand_normalizer'),
            normalize=config.data.normalize, min_max=config.data.min_max, rot_rep=config.data.rot_rep,
            device=args.device)
        denormalize_fn = Normalizer.offline_denormalize
        save_renders_local = partial(multiple_render, part='hand', bg_img=np.ones([192 * 2, 256 * 2, 3]) * 255,
                                     focal=[5000 * 2, 5000 * 2],
                                     princpt=[80 * 2, 128 * 2], device=args.device)
        lhand_samples, rhand_samples = samples[:, 63:63 + 45], samples[:, 63 + 45:63 + 90]
        body_model = MANO(model_path='../body_models/mano', batch_size=sample_num).to(args.device)
        save_renders_local(lhand_samples, denormalize_fn, body_model, target_path, 'generated_lhand_sample{}.png',
                           faster=args.faster, view='half_right_bottom')
        save_renders_local(rhand_samples, denormalize_fn, body_model, target_path, 'generated_rhand_sample{}.png',
                           faster=args.faster, view='half_right_bottom')
        print(f'hand samples saved under {target_path}')

        return

    elif args.task == 'eval_generation':
        '''
        evaluate APD and SI
        '''

        sample_num = 50000
        sampling_shape = (sample_num, DATA_DIM)
        sampling_eps = 5e-3
        default_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                                   device=args.device)
        _, samples = default_sampler(model, observation=None)
        samples = denormalize_fn(samples, to_axis=True)

        data_root = os.path.join(args.data_path, 'wholebody_data')
        fid = evaluate_fid(samples, f'{data_root}/statistics.npz')
        print('FID for 50000 generated samples', fid)
        prdc = evaluate_prdc(samples, f'{data_root}/reference_batch.pt')
        print(prdc)
        if os.path.exists(f'{data_root}/faiss_index.bin'):
            dnn = evaluate_dnn(samples, f'{data_root}/all_data.pt',
                               measure='absolute', faiss_index_path=f'{data_root}/faiss_index.bin')
            print('DNN for 50000 generated samples', dnn)
        samples = samples[:500]
        body_model = BodyModel(bm_path=args.bodymodel_path, num_betas=10, model_type='smplx',
                               batch_size=500).to(args.device)
        body_out = body_model(wholebody_params=samples)
        joints3d = body_out.Jtr
        APD = average_pairwise_distance_wholebody(joints3d)
        print('average_pairwise_distance (body) for 500 generated samples', APD)

        left_hand_pose = samples[:, 63:63 + 45]
        right_hand_pose = samples[:, 63 + 45:63 + 90]
        lhand_model = MANO(model_path='../body_models/mano', is_rhand=False,
                           batch_size=500).to(args.device)
        rhand_model = MANO(model_path='../body_models/mano', is_rhand=True,
                           batch_size=500).to(args.device)
        lhand_joint3d = lhand_model(hand_pose=left_hand_pose).OpJtr  # 21 joints
        rhand_joint3d = rhand_model(hand_pose=right_hand_pose).OpJtr  # 21 joints
        APD = (average_pairwise_distance(lhand_joint3d) + average_pairwise_distance(rhand_joint3d)) / 2
        print('average_pairwise_distance (hands) for 500 generated samples', APD)
        return

    """
    *****************        data preparation for demo tasks       *****************    
    """
    data = np.load(args.file_path, allow_pickle=True)
    sample_num = 10
    body_poses = np.concatenate([data['body_pose'], data['left_hand_pose'], data['right_hand_pose'],
                                 data['jaw_pose'], data['expression']], axis=1)[:sample_num]
    print(f'loaded axis pose data {body_poses.shape} from {args.file_path}')
    body_poses = torch.from_numpy(body_poses).to(args.device)
    body_model = BodyModel(bm_path=args.bodymodel_path, num_betas=10, model_type='smplx',
                           batch_size=sample_num).to(args.device)

    if args.task == 'view':
        target_path = os.path.join(args.output_path, 'view')
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        save_renders(body_poses, None, body_model, target_path, 'GT_sample{}.png',
                     convert=False, faster=args.faster, view=args.view)
        print(f'rendered images saved under {target_path}')
        return
    elif args.task == 'completion':
        # Perform completion baselines (ScoreSDE, MCG, DPS)
        task_args = SimpleNamespace(task=None)
        if args.mode == 'DPS':
            task_args.task, inverse_solver = 'default', 'BP'
        elif args.mode == 'DSG':
            task_args.task, inverse_solver = 'default', 'ABP'
        elif args.mode == 'MCG':
            task_args.task, inverse_solver = 'completion', 'BP'
        elif args.mode == 'ScoreSDE':
            task_args.task, inverse_solver = 'completion', None
        else:  # plain generation sampler
            task_args.task, inverse_solver = 'default', None
        comp_sampler = sampling.get_sampling_fn(config, sde, body_poses.shape, inverse_scaler, sampling_eps,
                                                device=args.device, inverse_solver=inverse_solver)

        comp_fn = DPoserComp(model, sde, config.training.continuous, improve_baseline=False)
        body_model = BodyModel(bm_path=args.bodymodel_path,
                               num_betas=10,
                               batch_size=sample_num,
                               model_type='smplx').to(args.device)

        gts = body_poses
        body_poses = normalizer_fn(body_poses, from_axis=True)  # [b, N_POSES*6]
        mask, observation, mask_type = create_wholebody_mask(body_poses, part=args.part, mask_root=True)

        hypo_num = args.hypo
        multihypo_denoise = []
        for hypo in range(hypo_num):
            if args.mode == 'DPoser':
                completion = comp_fn.optimize(observation, mask)
            else:
                _, completion = comp_sampler(model, observation=observation, mask=mask, args=task_args)
            multihypo_denoise.append(completion)
        multihypo_denoise = torch.stack(multihypo_denoise, dim=1)

        preds = denormalize_fn(multihypo_denoise, to_axis=True)

        evaler = Evaler(body_model=body_model, part=args.part)
        eval_results = evaler.multi_eval_bodys_all(preds, gts)
        evaler.print_multi_eval_result_all(eval_results, hypo_num=hypo_num)

        Normalizer = Posenormalizer(
            data_path=os.path.join(args.data_path, 'hand_data', 'hand_normalizer'),
            normalize=config.data.normalize, min_max=config.data.min_max, rot_rep=config.data.rot_rep,
            device=args.device)
        hand_denormalize_fn = Normalizer.offline_denormalize
        save_renders_local = partial(multiple_render, part='hand', bg_img=np.ones([192 * 2, 256 * 2, 3]) * 255,
                                     focal=[5000 * 2, 5000 * 2],
                                     princpt=[80 * 2, 128 * 2], device=args.device)

        target_path = os.path.join(args.output_path, 'completion', args.part)
        save_renders(gts, denormalize_fn, body_model, target_path, 'sample{}_original.png', convert=False,
                     faster=args.faster, view=args.view)
        print(f'Original samples under {target_path}')
        lhand_samples, rhand_samples = gts[:, 63:63 + 45], gts[:, 63 + 45:63 + 90]
        lhand_samples[..., 1::3] *= -1
        lhand_samples[..., 2::3] *= -1
        hand_model = MANO(model_path='../body_models/mano', batch_size=sample_num).to(args.device)
        save_renders_local(lhand_samples, None, hand_model, target_path,
                           'sample{}_original_lhand.png', faster=args.faster, convert=False,
                           view='half_right_bottom')
        save_renders_local(rhand_samples, None, hand_model, target_path,
                           'sample{}_original_rhand.png', faster=args.faster, convert=False,
                           view='half_right_bottom')
        print(f'GT hand samples saved under {target_path}')

        save_renders(observation, denormalize_fn, body_model, target_path, 'sample{}_masked.png', faster=args.faster,
                     view=args.view)
        print(f'Masked samples under {target_path}')

        for hypo in range(hypo_num):
            save_renders(multihypo_denoise[:, hypo], denormalize_fn, body_model, target_path,
                         'sample{}_completion' + str(hypo) + '.png', faster=args.faster, view=args.view)
        print(f'Completion samples under {target_path}')

        for hypo in range(hypo_num):
            samples = multihypo_denoise[:, hypo]
            lhand_samples, rhand_samples = samples[:, 63:63 + 45], samples[:, 63 + 45:63 + 90]
            save_renders_local(lhand_samples, hand_denormalize_fn, hand_model, target_path,
                               'sample{}_completion_lhand' + str(hypo) + '.png', faster=args.faster,
                               view='half_right_bottom')
            save_renders_local(rhand_samples, hand_denormalize_fn, hand_model, target_path,
                               'sample{}_completion_rhand' + str(hypo) + '.png', faster=args.faster,
                               view='half_right_bottom')
        print(f'Completion hand samples saved under {target_path}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    app.run(main, flags_parser=parse_args)