import argparse
import json
import os.path
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.body_model import constants
from lib.body_model.body_model import BodyModel, fullpose_to_params
from lib.body_model.fitting_losses import perspective_projection, guess_init
from lib.body_model.joint_mapping import mmpose_to_openpose, vitpose_to_openpose
from lib.body_model.visual import Renderer, vis_keypoints_with_skeleton
from lib.dataset.mocap_dataset import MocapDataset
from lib.utils.preprocess import compute_bbox
from lib.utils.transforms import cam_crop2full
from .smplify import SMPLify


parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='DPoser', choices=['DPoser', 'None'],)
parser.add_argument('--config-path', type=str,
                    default='configs.wholebody.subvp.finetune_mixed.get_config', help='config files to build DPoser')
parser.add_argument('--ckpt-path', type=str,
                    default='./pretrained_models/wholebody/mixed/last.ckpt',
                    help='load trained diffusion model')

parser.add_argument('--data-path', type=str, default='./data', help='normalizer folder')
parser.add_argument('--bodymodel-path', type=str, default='../body_models/smplx',
                    help='path of SMPLX model')

parser.add_argument('--time-strategy', type=str, default='3', choices=['1', '2', '3'],
                    help='random, fix, truncated annealing')

parser.add_argument('--data_dir', type=str,
                    default='../data/human/WholeBodydataset/Arctic/data', help='Path to data directory')
parser.add_argument('--kpts', type=str, default='mmpose', choices=['mmpose'])
parser.add_argument('--init_camera', type=str, default='fixed', choices=['fixed', 'optimized'])
parser.add_argument('--outdir', type=str, default='./output/wholebody/test_results/hmr_arctic/gt_kpts',
                    help='output directory of fitting visualization results')
parser.add_argument('--interpenetration', '-i', action='store_true', help='enable interpenetration penalty')
parser.add_argument('--gt-intrinsic', action='store_true', )
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--device', type=str, default='cuda:0')


if __name__ == '__main__':
    torch.manual_seed(42)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device
    fitting = True
    enable_visual = False

    batch_size = args.batch_size
    step_size, num_iters = 1e-2, 100

    with open(f"{args.data_dir}/meta/misc.json", "r", ) as f:
        misc = json.load(f)

    statcams = {}
    for sub in misc.keys():
        statcams[sub] = {
            "world2cam": torch.FloatTensor(np.array(misc[sub]["world2cam"])),
            "intris_mat": torch.FloatTensor(np.array(misc[sub]["intris_mat"])),
        }

    # read all actions of a person
    person_id = 's05'
    camera_id = 4
    data_dir = args.data_dir
    person_dir = f"{data_dir}/images/{person_id}"
    actions = sorted(d for d in os.listdir(person_dir) if os.path.isdir(os.path.join(person_dir, d)))

    all_img_paths = []
    all_json_paths = []
    all_gt_verts = []

    # sample every 20th frame
    indices = slice(19, None, 20)

    for idx, action in enumerate(actions):
        img_paths = sorted(glob(f"{person_dir}/{action}/{camera_id}/*.jpg"))[1:][indices]
        all_img_paths.extend(img_paths)
        print(f"Processing {person_id} {action} {camera_id}, ({idx + 1}/{len(actions)})")
        gt_dict_path = f"{data_dir}/processed_verts/seqs/{person_id}/{action}.npy"
        gt_data = np.load(gt_dict_path, allow_pickle=True, encoding='latin1').item()
        gt_verts = gt_data['cam_coord']['verts.smplx'][:, camera_id][indices]
        all_gt_verts.append(gt_verts)

        if args.kpts == 'mmpose':
            json_paths = sorted(glob(f"{data_dir}/keypoints/mmpose_keypoints/{person_id}/{action}/{camera_id}/*.json"))[1:][indices]
        else:
            raise NotImplementedError
        all_json_paths.extend(json_paths)
    all_gt_verts = np.concatenate(all_gt_verts, axis=0)

    # Load SMPLX model
    smpl = BodyModel(bm_path=args.bodymodel_path, num_betas=10, model_type='smplx', batch_size=batch_size).to(device)
    N_POSES = 22

    img_names = []
    for img_path in all_img_paths:
        # Split the path and get the last 4 components
        parts = img_path.split('/')[-4:]
        # Join the relevant parts and remove the file extension
        img_name = '_'.join(parts[1:]).rsplit('.', 1)[0]
        img_names.append(img_name)
    total_length = len(all_img_paths)
    current_idx, batch_keypoints, batch_img = 0, [], []
    all_eval_results = {}

    for img_path, json_path in tqdm(zip(all_img_paths, all_json_paths), desc='Dataset', total=total_length):
        base_name = os.path.basename(img_path)
        img_name, _ = os.path.splitext(base_name)
        # load image and 2D keypoints
        img_bgr = cv2.imread(img_path)
        json_data = json.load(open(json_path))
        if args.kpts == 'mmpose':
            mm_keypoints = np.array(json_data[0]['keypoints'])
            keypoint_scores = np.array(json_data[0]['keypoint_scores'])
            keypoints = mmpose_to_openpose(mm_keypoints, keypoint_scores)
        else:
            raise NotImplementedError('Unknown keypoints type')
        batch_keypoints.append(keypoints)
        batch_img.append(img_bgr)

        if len(batch_keypoints) < batch_size:
            continue

        bboxes = compute_bbox(batch_keypoints)
        keypoints = np.array(batch_keypoints)
        print('batch keypoints:', keypoints.shape)

        assert len(bboxes) == batch_size
        mocap_db = MocapDataset(batch_img, bboxes, batch_size, args.device, body_model_path=args.bodymodel_path)
        mocap_data_loader = DataLoader(mocap_db, batch_size=batch_size, num_workers=0)

        for batch in mocap_data_loader:
            img_h = batch["img_h"].to(device).float()
            img_w = batch["img_w"].to(device).float()
            if args.gt_intrinsic:
                selected_intris_mat = statcams[person_id]["intris_mat"][camera_id - 1]
                focal_length = torch.tensor(selected_intris_mat[0, 0]).to(device).float().repeat(batch_size)
                camera_center = torch.tensor(selected_intris_mat[:2, 2]).to(device).float().repeat(batch_size, 1)
            else:
                focal_length = batch["focal_length"].to(device).float()
                camera_center = torch.hstack((img_w[:, None], img_h[:, None])) / 2

            keypoints_tensor = torch.from_numpy(keypoints).to(device)

            smpl_poses = smpl.mean_poses.unsqueeze(0).repeat(batch_size, 1).to(device)  # [N, 165]
            grab_pose = torch.from_numpy(np.load(constants.GRAB_POSE_PATH)['pose'][:, :N_POSES * 3]).to(
                smpl_poses.device)
            smpl_poses[:, :N_POSES * 3] = grab_pose[:, :N_POSES * 3]
            init_betas = smpl.mean_shape.unsqueeze(0).repeat(batch_size, 1).to(device)  # N*10

            # Convert the camera parameters from the crop camera to the full camera
            if args.init_camera == 'fixed':
                center = batch["center"].to(device).float()
                scale = batch["scale"].to(device).float()
                full_img_shape = torch.stack((img_h, img_w), dim=-1)
                pred_cam_crop = torch.tensor([[1.1, 0, 0]], device=device).repeat(batch_size, 1)
                init_cam_t = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)
            else:
                init_joints_3d = smpl(betas=init_betas,
                                      body_pose=smpl_poses[:, 3:N_POSES * 3],
                                      global_orient=smpl_poses[:, :3], ).Jtr
                init_cam_t = guess_init(init_joints_3d[:, :25], keypoints_tensor[:, :25], focal_length, part='body')

            init_output = smpl(betas=init_betas,
                               body_pose=smpl_poses[:, 3:N_POSES * 3],
                               global_orient=smpl_poses[:, :3],
                               trans=init_cam_t)

            # be careful: the estimated focal_length should be used here instead of the default constant
            smplify = SMPLify(body_model=smpl, step_size=step_size, batch_size=batch_size, num_iters=num_iters,
                                    focal_length=focal_length, args=args)
            if fitting:
                results = smplify(init_output.full_pose.detach(),
                                  init_output.expression.detach(),
                                  init_output.betas.detach(),
                                  init_cam_t.detach(),
                                  camera_center,
                                  keypoints_tensor)
                global_orient, wholebody_params, betas, cam_t, joint_loss = results
            else:
                global_orient = init_output.full_pose[:, :3]
                wholebody_params = torch.cat([fullpose_to_params(init_output.full_pose), init_output.expression], dim=-1)
                betas = init_output.betas
                cam_t = init_cam_t

            fitted_output = smpl(betas=betas,
                                 wholebody_params=wholebody_params,
                                 global_orient=global_orient,
                                 trans=cam_t)
            pred_vertices = fitted_output.v

            batch_results = mocap_db.eval_arctic_wholebody(pred_vertices, all_gt_verts[current_idx: current_idx + batch_size])
            for key, value in batch_results.items():
                if key not in all_eval_results:
                    all_eval_results[key] = []
                all_eval_results[key].extend(value)
            if enable_visual:
                # re-project to 2D keypoints on image plane
                pred_keypoints3d = fitted_output.OpJtr
                rotation = torch.eye(3, device=device).unsqueeze(0).expand(pred_keypoints3d.shape[0], -1, -1)
                projected_joints = perspective_projection(pred_keypoints3d, rotation, cam_t,
                                                          focal_length, camera_center).detach().cpu().numpy()
                dummy_confidence = np.ones((projected_joints.shape[0], projected_joints.shape[1], 1))
                projected_joints = np.concatenate([projected_joints, dummy_confidence], axis=-1)

                front_view_list = []
                batch_img_rgb = [img[:, :, ::-1] for img in batch_img]
                for idx, img_name in enumerate(img_names[current_idx: current_idx + batch_size]):
                    img_with_kpts = vis_keypoints_with_skeleton(batch_img[idx], keypoints[idx], kp_thresh=0.3, radius=6)
                    cv2.imwrite(os.path.join(args.outdir, f"{img_name}_kpt2d_gt.jpg"), img_with_kpts)

                    img_with_kpts = vis_keypoints_with_skeleton(batch_img[idx], projected_joints[idx], kp_thresh=0.3, radius=6)
                    cv2.imwrite(os.path.join(args.outdir, f"{img_name}_kpt2d_pred.jpg"), img_with_kpts)

                    # visualize predicted mesh
                    renderer = Renderer(focal_length=focal_length[idx], img_w=img_w[idx], img_h=img_h[idx],
                                        camera_center=camera_center[idx], faces=smpl.faces, same_mesh_color=True)
                    front_view = renderer.render_front_view(pred_vertices[idx:idx+1].detach().cpu().numpy(),
                                                            batch_img_rgb[idx].copy())
                    front_view_list.append(front_view)
                    renderer.delete()

                # visualize gt mesh
                selected_intris_mat = statcams[person_id]["intris_mat"][camera_id - 1]
                gt_focal_length = torch.tensor(selected_intris_mat[0, 0]).to(device).float()
                gt_camera_center = torch.tensor(selected_intris_mat[:2, 2]).to(device).float()
                renderer = Renderer(focal_length=gt_focal_length, img_w=img_w[0], img_h=img_h[0],
                                    camera_center=gt_camera_center, faces=smpl.faces, same_mesh_color=True)
                gt_front_view_list = renderer.render_multiple_front_view(all_gt_verts[current_idx: current_idx + batch_size],
                                                                        [img.copy() for img in batch_img_rgb])
                renderer.delete()
                for vis_idx, img_name in enumerate(img_names[current_idx: current_idx + batch_size]):
                    cv2.imwrite(os.path.join(args.outdir, f"{img_name}_mesh_fit.jpg"), front_view_list[vis_idx][:, :, ::-1])
                    cv2.imwrite(os.path.join(args.outdir, f"{img_name}_mesh_gt.jpg"), gt_front_view_list[vis_idx][:, :, ::-1])

        batch_keypoints, batch_img = [], []  # clear for the next batch
        current_idx += batch_size

    print('results on whole dataset:')
    mocap_db.print_eval_result_wholebody(all_eval_results)