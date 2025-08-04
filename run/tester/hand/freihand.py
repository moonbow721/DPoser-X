import argparse
import json
import os
from tqdm import tqdm

import cv2
import numpy as np
import torch

from lib.body_model.fitting_losses import guess_init, perspective_projection
from lib.body_model.hand_model import MANO
from lib.utils.preprocess import get_best_hand
from lib.utils.transforms import get_focal_pp, rigid_align
from lib.body_model.visual import Renderer, vis_keypoints_with_skeleton
from .smplify import DPoser, SMPLify

parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='DPoser', choices=['DPoser', 'None'],)
parser.add_argument('--ckpt-path', type=str,
                    default='./pretrained_models/hand/BaseMLP/last.ckpt',
                    help='load trained diffusion model for DPoser')
parser.add_argument('--config-path', type=str,
                    default='configs.hand.subvp.timefc.get_config',
                    help='config files to build DPoser')

parser.add_argument('--data-path', type=str, default='./data/hand_data')
parser.add_argument('--bodymodel-path', type=str, default='../body_models/mano')

parser.add_argument('--time-strategy', type=str, default='3', choices=['1', '2', '3'],
                    help='random, fix, truncated annealing')

parser.add_argument('--data-dir', type=str, default='../data/human/Handdataset/FreiHAND/evaluation')
parser.add_argument('--kpts', type=str, default='mmpose_hand', choices=['gt', 'mmpose_hand'])
parser.add_argument('--init', type=str, default='none', choices=['none', 'hand4whole'])
parser.add_argument('--outdir', type=str, default='./output/hand/test_results/hmr',)
parser.add_argument('--device', type=str, default='cuda:0')


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device
    fitting = True
    enable_visual = False
    render_gt = True
    logging = True

    mano = MANO(model_path=args.bodymodel_path, batch_size=1).to(device)
    N_POSES = 16  # including root orient
    batch_size = 1

    rgb_dir = os.path.join(args.data_dir, 'rgb')
    img_paths = sorted([os.path.join(rgb_dir, x) for x in os.listdir(rgb_dir)])

    print('Total images:', len(img_paths))
    summary_results = {"img_name": [], "reproj_err": [], "verts_err": [], "joints_err": [], "joints": [], "verts": []}

    # build Pose prior
    if args.prior == 'DPoser':
        pose_prior = DPoser(batch_size, args.config_path, args)
    else:
        pose_prior = None

    for img_path in tqdm(img_paths, desc='Processing'):
        base_name = os.path.basename(img_path).split('.')[0]
        summary_results["img_name"].append(os.path.basename(img_path))
        # load image
        img_bgr = cv2.imread(img_path)
        img_h = torch.tensor(img_bgr.shape[0], device=device).float()
        img_w = torch.tensor(img_bgr.shape[1], device=device).float()

        # load keypoints
        if args.kpts == 'gt':
            kpt_path = os.path.join(args.data_dir, 'gt_keypoints', f'{base_name}.json')
            kpts = np.array(json.load(open(kpt_path))['hand_keypoints'])
        elif args.kpts == 'mmpose_hand':
            kpt_path = os.path.join(args.data_dir, 'mmpose_hand', 'predictions', f'{base_name}.json')
            json_data = json.load(open(kpt_path))
            person_num = len(json_data)
            keypoints_list = []
            for i in range(person_num):
                keypoints = np.array(json_data[i]['keypoints'])
                keypoint_scores = np.array(json_data[i]['keypoint_scores'])
                converted_keypoints = np.hstack((keypoints, keypoint_scores.reshape(-1, 1)))
                keypoints_list.append(converted_keypoints)
            best_idx = get_best_hand(keypoints_list, hand='rhand', from_wholebody=False)
            kpts = keypoints_list[best_idx]  # [21, 3]
        else:
            raise ValueError('Unknown keypoint source')
        keypoints = torch.from_numpy(kpts[np.newaxis, ...]).to(device).float()

        # load annotations
        anno_path = os.path.join(args.data_dir, 'anno', f'{base_name}.json')
        anno_data = json.load(open(anno_path))
        K, mano_params = [torch.tensor(x, device=device).float() for x in [anno_data['K'], anno_data['mano'],]]
        joint_gt, mesh_gt = [np.array(x) for x in [anno_data['xyz'], anno_data['verts']]]

        if args.init == 'none':
            focal_length, camera_center = get_focal_pp(K)
            camera_center = camera_center[None,].float()
            smpl_poses = mano.mean_poses[:N_POSES * 3].repeat(batch_size, 1).to(device)
            init_betas = mano.mean_shape.repeat(batch_size, 1).to(device)

            init_joints_3d = mano(betas=init_betas,
                                  hand_pose=smpl_poses[:, 3:],
                                  global_orient=smpl_poses[:, :3]).OpJtr
            init_cam_t = guess_init(init_joints_3d, keypoints, focal_length, part='rhand')
            # init_cam_t[:, 2] *= 0.8  # rescale estimated depth
        elif args.init == 'hand4whole':
            results_path = os.path.join(args.data_dir, 'pose2pose', 'predictions', f'{base_name}.json')
            results = json.load(open(results_path))
            smpl_poses = torch.tensor(results['pose'], device=device)[None,].float()
            init_betas = torch.tensor(results['shape'], device=device)[None,].float()
            init_cam_t = torch.tensor(results['cam_trans'], device=device)[None,].float()
            camera_center = torch.tensor(results['princpt'], device=device)[None,].float()
            focal_length = torch.tensor(results['focal'][0], device=device).float()
        else:
            raise ValueError('Unknown initialization strategy')

        # be careful: the estimated focal_length should be used here instead of the default constant
        smplify = SMPLify(body_model=mano, step_size=5e-2, batch_size=batch_size, num_iters=100,
                          focal_length=focal_length, args=args)
        smplify.load_prior(prior=pose_prior)

        if fitting:
            results = smplify(smpl_poses.detach(),
                              init_betas.detach(),
                              init_cam_t.detach(),
                              camera_center,
                              keypoints)
            fitted_pose, fitted_shape, fitted_trans, repro_loss = results
        else:
            fitted_pose, fitted_shape, fitted_trans, repro_loss = [smpl_poses, init_betas, init_cam_t, np.array([0.0])]

        print('after re-projection loss', repro_loss.mean().item())
        summary_results["reproj_err"].append(repro_loss.mean().item())

        with torch.no_grad():
            pred_output = mano(betas=fitted_shape,
                               hand_pose=fitted_pose[:, 3:],
                               global_orient=fitted_pose[:, :3],
                               transl=fitted_trans)
            pred_vertices = pred_output.v
            pred_keypoints3d = pred_output.OpJtr

        summary_results["joints"].append(pred_keypoints3d[0].cpu().numpy().tolist())
        summary_results["verts"].append(pred_vertices[0].cpu().numpy().tolist())

        pred_vert_aligned = rigid_align(pred_vertices[0].cpu().numpy(), mesh_gt)
        pred_joint_aligned = rigid_align(pred_keypoints3d[0].cpu().numpy(), joint_gt)
        verts_err = np.sqrt(np.sum((pred_vert_aligned - mesh_gt) ** 2, 1)).mean() * 1000
        joints_err = np.sqrt(np.sum((pred_joint_aligned - joint_gt) ** 2, 1)).mean() * 1000
        print('aligned v2v loss:', verts_err)
        print('aligned j2j loss:', joints_err)
        summary_results["verts_err"].append(verts_err.item())
        summary_results["joints_err"].append(joints_err.item())

        if enable_visual:
            os.makedirs(os.path.join(args.outdir, "visual_dir"), exist_ok=True)
            img_with_kpts = vis_keypoints_with_skeleton(img_bgr, kpts, kp_thresh=0.1, radius=2)

            cv2.imwrite(os.path.join(args.outdir, "visual_dir", f"{base_name}_kpt2d_gt.jpg"), img_with_kpts)
            rotation = torch.eye(3, device=device).unsqueeze(0).expand(pred_keypoints3d.shape[0], -1, -1)
            projected_joints = perspective_projection(pred_keypoints3d, rotation, fitted_trans,
                                                      focal_length, camera_center).detach().cpu().numpy()
            dummy_confidence = np.ones((projected_joints.shape[0], projected_joints.shape[1], 1))
            projected_joints = np.concatenate([projected_joints, dummy_confidence], axis=-1)
            img_with_kpts = vis_keypoints_with_skeleton(img_bgr, projected_joints[0], kp_thresh=0.1, radius=2)
            cv2.imwrite(os.path.join(args.outdir, "visual_dir", f"{base_name}_kpt2d_pred.jpg"), img_with_kpts)

            renderer = Renderer(focal_length=focal_length, camera_center=camera_center[0], img_w=img_w, img_h=img_h,
                                faces=mano.faces, same_mesh_color=True)
            front_view = renderer.render_front_view(pred_vertices.cpu().numpy(), bg_img_rgb=img_bgr[:, :, ::-1].copy())
            side_view = renderer.render_side_view(pred_vertices.cpu().numpy())
            renderer.delete()

            cv2.imwrite(os.path.join(args.outdir, "visual_dir", f"{base_name}_mesh_fit.jpg"), front_view[:, :, ::-1])
            cv2.imwrite(os.path.join(args.outdir, "visual_dir", f"{base_name}_mesh_fit_side.jpg"), side_view[:, :, ::-1])

            if render_gt:
                renderer = Renderer(focal_length=get_focal_pp(K)[0], camera_center=get_focal_pp(K)[1], img_w=img_w, img_h=img_h,
                                    faces=mano.faces, same_mesh_color=True)
                front_view = renderer.render_front_view(mesh_gt[np.newaxis, ...], bg_img_rgb=img_bgr[:, :, ::-1].copy())
                side_view = renderer.render_side_view(mesh_gt[np.newaxis, ...])
                renderer.delete()

                cv2.imwrite(os.path.join(args.outdir, "visual_dir", f"{base_name}_mesh_gt.jpg"), front_view[:, :, ::-1])
                cv2.imwrite(os.path.join(args.outdir, "visual_dir", f"{base_name}_mesh_gt_side.jpg"), side_view[:, :, ::-1])

    # print mean results
    print('Mean reprojection error:', np.mean(summary_results["reproj_err"]))
    print('Mean vertex error:', np.mean(summary_results["verts_err"]))
    print('Mean joint error:', np.mean(summary_results["joints_err"]))

    if logging:
        # save as json file to args.outpath
        with open(os.path.join(args.outdir, 'pred.json'), 'w') as f:
            json.dump([summary_results["joints"], summary_results["verts"]], f)
        print('Results saved to', os.path.join(args.outdir, 'pred.json'))

        # save a light version of the results
        light_results = {"img_name": [], "reproj_err": [], "verts_err": [], "joints_err": []}
        for k, v in summary_results.items():
            light_results[k] = v
        light_results.pop('joints')
        light_results.pop('verts')
        with open(os.path.join(args.outdir, 'light_results.json'), 'w') as f:
            json.dump(light_results, f)
        print('Light results saved to', os.path.join(args.outdir, 'light_results.json'))
