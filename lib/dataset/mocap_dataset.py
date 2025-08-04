# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
import pickle

import cv2
import numpy as np

from torch.utils.data import Dataset
from lib.body_model.body_model import BodyModel
from lib.utils.preprocess import process_image, load_ply
from lib.utils.transforms import estimate_focal_length, rigid_align


class MocapDataset(Dataset):
    def __init__(self, img_bgr_list, detection_list, batchsize=1, device='cuda:0', body_model_path=None):
        self.img_bgr_list = img_bgr_list
        self.detection_list = detection_list
        self.device = device

        # To evaluate EHF
        self.cam_param = {'R': [-2.98747896, 0.01172457, -0.05704687]}
        self.cam_param['R'], _ = cv2.Rodrigues(np.array(self.cam_param['R']))
        if body_model_path is not None:
            self.smplx = BodyModel(bm_path=body_model_path,
                                   num_betas=10,
                                   batch_size=batchsize,
                                   model_type='smplx').to(device)


    def __len__(self):
        return len(self.detection_list)

    def __getitem__(self, idx):
        """
        bbox: [batch_id, min_x, min_y, max_x, max_y]
        :param idx:
        :return:
        """
        item = {}
        img_idx = int(self.detection_list[idx][0].item())
        img_bgr = self.img_bgr_list[img_idx]
        img_rgb = img_bgr[:, :, ::-1]
        img_h, img_w, _ = img_rgb.shape
        focal_length = estimate_focal_length(img_h, img_w)

        bbox = self.detection_list[idx][1:5]
        norm_img, center, scale, crop_ul, crop_br, _ = process_image(img_rgb, bbox)

        item["norm_img"] = norm_img
        item["center"] = center
        item["scale"] = scale
        item["crop_ul"] = crop_ul
        item["crop_br"] = crop_br
        item["img_h"] = img_h
        item["img_w"] = img_w
        item["focal_length"] = focal_length
        return item

    def eval_EHF(self, pred_results, gt_ply_path):
        eval_result = {'pa_mpjpe_body': [], 'mpjpe_body': []}
        batchsize = pred_results[0].shape[0]
        if batchsize > 1:
            assert isinstance(gt_ply_path, list) and len(gt_ply_path) == batchsize
            gt_ply_path_list = gt_ply_path
        else:
            gt_ply_path_list = [gt_ply_path]
        pose, betas, camera_translation, reprojection_loss = pred_results
        mesh_out = self.smplx(betas=betas,
                              body_pose=pose[:, 3:66],
                              global_orient=pose[:, :3],
                              trans=camera_translation).v.detach().cpu().numpy()
        for idx, gt_ply in enumerate(gt_ply_path_list):
            mesh_gt = load_ply(gt_ply)
            mesh_gt = np.dot(self.cam_param['R'], mesh_gt.transpose(1, 0)).transpose(1, 0)
            # MPJPE from body joints
            joint_gt_body = np.dot(self.smplx.J_regressor, mesh_gt)[:22]
            joint_out_body = np.dot(self.smplx.J_regressor, mesh_out[idx])[:22]
            joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
            eval_result['pa_mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, axis=1)).mean() * 1000)
            joint_out_body_align = joint_out_body - joint_out_body[self.smplx.J_regressor_idx['pelvis'], None, :] + \
                                   joint_gt_body[self.smplx.J_regressor_idx['pelvis'], None, :]
            eval_result['mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, axis=1)).mean() * 1000)

        return eval_result

    def eval_arctic_wholebody(self, pred_vertices, gt_vertices):
        eval_result = {
            'pa_mpvpe_all': [], 'mpvpe_all': [],
            'pa_mpvpe_hand': [], 'mpvpe_hand': [],
            'pa_mpvpe_face': [], 'mpvpe_face': [],
            'pa_mpjpe_body': [], 'pa_mpjpe_hand': []
        }
        batchsize = pred_vertices.shape[0]
        mesh_out_all = pred_vertices.detach().cpu().numpy()
        for idx in range(batchsize):
            mesh_gt = gt_vertices[idx]

            # MPVPE from all vertices
            mesh_out = mesh_out_all[idx]
            mesh_out_align = rigid_align(mesh_out, mesh_gt)
            eval_result['pa_mpvpe_all'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000)
            mesh_out_align = mesh_out - np.dot(self.smplx.J_regressor, mesh_out)[self.smplx.J_regressor_idx['pelvis'],
                                        None,
                                        :] + np.dot(self.smplx.J_regressor, mesh_gt)[
                                             self.smplx.J_regressor_idx['pelvis'], None,
                                             :]
            eval_result['mpvpe_all'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1)).mean() * 1000)

            # MPVPE from hand vertices
            mesh_gt_lhand = mesh_gt[self.smplx.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand = mesh_out[self.smplx.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
            mesh_gt_rhand = mesh_gt[self.smplx.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand = mesh_out[self.smplx.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)
            eval_result['pa_mpvpe_hand'].append((np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

            mesh_out_lhand_align = mesh_out_lhand - np.dot(self.smplx.J_regressor, mesh_out)[
                                                    self.smplx.J_regressor_idx['lwrist'], None, :] + np.dot(
                self.smplx.J_regressor, mesh_gt)[self.smplx.J_regressor_idx['lwrist'], None, :]
            mesh_out_rhand_align = mesh_out_rhand - np.dot(self.smplx.J_regressor, mesh_out)[
                                                    self.smplx.J_regressor_idx['rwrist'], None, :] + np.dot(
                self.smplx.J_regressor, mesh_gt)[self.smplx.J_regressor_idx['rwrist'], None, :]
            eval_result['mpvpe_hand'].append((np.sqrt(
                np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

            # MPVPE from face vertices
            mesh_gt_face = mesh_gt[self.smplx.face_vertex_idx, :]
            mesh_out_face = mesh_out[self.smplx.face_vertex_idx, :]
            mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
            eval_result['pa_mpvpe_face'].append(
                np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)
            mesh_out_face_align = mesh_out_face - np.dot(self.smplx.J_regressor, mesh_out)[
                                                  self.smplx.J_regressor_idx['neck'],
                                                  None, :] + np.dot(self.smplx.J_regressor, mesh_gt)[
                                                             self.smplx.J_regressor_idx['neck'], None, :]
            eval_result['mpvpe_face'].append(
                np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)).mean() * 1000)

            # MPJPE from body joints
            joint_gt_body = np.dot(self.smplx.j14_regressor, mesh_gt)
            joint_out_body = np.dot(self.smplx.j14_regressor, mesh_out)
            joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
            eval_result['pa_mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, 1)).mean() * 1000)

            # MPJPE from hand joints
            joint_gt_lhand = np.dot(self.smplx.orig_hand_regressor['left'], mesh_gt)
            joint_out_lhand = np.dot(self.smplx.orig_hand_regressor['left'], mesh_out)
            joint_out_lhand_align = rigid_align(joint_out_lhand, joint_gt_lhand)
            joint_gt_rhand = np.dot(self.smplx.orig_hand_regressor['right'], mesh_gt)
            joint_out_rhand = np.dot(self.smplx.orig_hand_regressor['right'], mesh_out)
            joint_out_rhand_align = rigid_align(joint_out_rhand, joint_gt_rhand)
            eval_result['pa_mpjpe_hand'].append((np.sqrt(
                np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)).mean() * 1000 + np.sqrt(
                np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)).mean() * 1000) / 2.)

        return eval_result

    def print_eval_result(self, eval_result):
        print('PA MPJPE (Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body']))
        print('MPJPE (Body): %.2f mm' % np.mean(eval_result['mpjpe_body']))
        if 'pck_body' in eval_result:
            print('PCK: %.5f mm' % np.mean(eval_result['pck_body']))

    def print_eval_result_wholebody(self, eval_result):
        for key in eval_result:
            print(f'{key}: {np.mean(eval_result[key]):.2f} mm')
