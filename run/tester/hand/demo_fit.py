import argparse
import json
import os.path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.body_model.fitting_losses import perspective_projection, guess_init
from lib.body_model.hand_model import MANO

from lib.body_model.visual import Renderer, vis_keypoints_with_skeleton
from lib.dataset.mocap_dataset import MocapDataset
from lib.utils.transforms import cam_crop2full
from .smplify import SMPLify, DPoser

parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='DPoser', choices=['DPoser', 'None'],
                    help='Our prior model or competitors')
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

parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--mmpose', type=str, required=True, help='Path to .json containing mmpose detections')
parser.add_argument('--outdir', type=str, default='./output/hand/test_results/hmr',
                    help='output directory of fitting visualization results')
parser.add_argument('--device', type=str, default='cuda:0')


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device

    # TODO: support left hand
    mano = MANO(model_path=args.bodymodel_path, batch_size=1).to(device)
    N_POSES = 16  # including root orient
    print("results will be saved under", args.outdir)

    if args.prior == 'DPoser':
        pose_prior = DPoser(1, args.config_path, args)
    else:
        pose_prior = None

    # load image and 2D keypoints from OpenPose
    orig_img_bgr_all = [cv2.imread(args.img)]
    json_data = json.load(open(args.mmpose))
    person_num = len(json_data)
    # TODO: support multiple instances and left hand in the future
    keypoints = np.array(json_data[0]['keypoints'])
    keypoint_scores = np.array(json_data[0]['keypoint_scores'])
    kpts = np.hstack((keypoints, keypoint_scores.reshape(-1, 1))).astype(np.float32)

    # # [batch_id, min_x, min_y, max_x, max_y]
    bboxes = []
    valid = kpts[:, 2] > 0.4
    if sum(valid) > 3:
        bbox = [np.array(0), kpts[valid, 0].min(), kpts[valid, 1].min(), kpts[valid, 0].max(), kpts[valid, 1].max()]
        bboxes.append(bbox)

    batch_size = len(bboxes)
    assert batch_size == 1, 'we only support single person and single image for this demo'

    mocap_db = MocapDataset(orig_img_bgr_all, bboxes, device=args.device)
    mocap_data_loader = DataLoader(mocap_db, batch_size=batch_size, num_workers=0)

    for batch in mocap_data_loader:
        img_h = batch["img_h"].to(device).float()
        img_w = batch["img_w"].to(device).float()
        focal_length = batch["focal_length"].to(device).float()

        keypoints = torch.from_numpy(kpts[np.newaxis, ...]).to(device)

        smpl_poses = mano.mean_poses[:N_POSES * 3].repeat(batch_size, 1).to(device)
        init_betas = mano.mean_shape.repeat(batch_size, 1).to(device)
        camera_center = torch.hstack((img_w[:, None], img_h[:, None])) / 2

        init_joints_3d = mano(betas=init_betas,
                              hand_pose=smpl_poses[:, 3:],
                              global_orient=smpl_poses[:, :3]).OpJtr
        init_cam_t = guess_init(init_joints_3d, keypoints, focal_length, part='rhand')

        # be careful: the estimated focal_length should be used here instead of the default constant
        smplify = SMPLify(body_model=mano, step_size=5e-2, batch_size=batch_size, num_iters=100,
                          focal_length=focal_length, args=args)
        smplify.load_prior(pose_prior)

        results = smplify(smpl_poses.detach(),
                          init_betas.detach(),
                          init_cam_t.detach(),
                          camera_center,
                          keypoints)
        new_opt_pose, new_opt_betas, new_opt_cam_t, new_opt_joint_loss = results
        print('after re-projection loss', new_opt_joint_loss.sum().item())

        with torch.no_grad():
            pred_output = mano(betas=new_opt_betas,
                               hand_pose=new_opt_pose[:, 3:],
                               global_orient=new_opt_pose[:, :3],
                               transl=new_opt_cam_t)
            pred_vertices = pred_output.v
            pred_keypoints3d = pred_output.OpJtr

        orig_img_bgr = orig_img_bgr_all[0].copy()
        img_with_kpts = vis_keypoints_with_skeleton(orig_img_bgr, kpts, kp_thresh=0.1, radius=2)
        cv2.imwrite(os.path.join(args.outdir, "kpt2d_gt.jpg"), img_with_kpts)
        rotation = torch.eye(3, device=device).unsqueeze(0).expand(pred_keypoints3d.shape[0], -1, -1)
        projected_joints = perspective_projection(pred_keypoints3d, rotation, new_opt_cam_t,
                                                  focal_length, camera_center).detach().cpu().numpy()
        dummy_confidence = np.ones((projected_joints.shape[0], projected_joints.shape[1], 1))
        projected_joints = np.concatenate([projected_joints, dummy_confidence], axis=-1)

        # remove batch dim
        img_with_kpts = vis_keypoints_with_skeleton(orig_img_bgr, projected_joints[0], kp_thresh=0.1, radius=2)
        cv2.imwrite(os.path.join(args.outdir, "kpt2d_pred.jpg"), img_with_kpts)

        # visualize predicted mesh
        renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                            faces=mano.faces, same_mesh_color=True)
        init_output = mano(betas=init_betas,
                           hand_pose=smpl_poses[:, 3:],
                           global_orient=smpl_poses[:, :3],
                           transl=init_cam_t)
        front_view = renderer.render_front_view(init_output.v.detach().cpu().numpy(),
                                                bg_img_rgb=orig_img_bgr_all[0][:, :, ::-1].copy())
        cv2.imwrite(os.path.join(args.outdir, "mesh_init.jpg"), front_view[:, :, ::-1])
        renderer.delete()

        renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                            faces=mano.faces, same_mesh_color=True)
        front_view = renderer.render_front_view(pred_vertices.cpu().numpy(),
                                                bg_img_rgb=orig_img_bgr_all[0][:, :, ::-1].copy())
        cv2.imwrite(os.path.join(args.outdir, "mesh_fit.jpg"), front_view[:, :, ::-1])
