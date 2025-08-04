import pickle

import numpy as np
import torch
import torch.nn as nn
from smplx import SMPL, SMPLH, SMPLX
from smplx.utils import Struct
import os

from lib.body_model import constants
from lib.body_model.joint_mapping import smpl_to_openpose
from lib.utils.transforms import rot6d_to_axis_angle

regressor_paths = ['/data3/ljz24/projects/3d/Hand4Whole/common/utils/human_model_files/smplx/SMPLX_to_J14.pkl',
                   '/data3/ljz24/projects/3d/body_models/smpl/J_regressor_h36m.npy']

def fullpose_to_params(fullpose):
    body_pose = fullpose[:, 3:3 + 63]
    jaw_pose = fullpose[:, 66:66 + 3]
    left_hand_pose = fullpose[:, 75:75 + 45]
    right_hand_pose = fullpose[:, 120:120 + 45]
    pose_params = torch.cat([body_pose, left_hand_pose, right_hand_pose, jaw_pose], dim=-1)
    return pose_params


class BodyModel(nn.Module):
    '''
    Wrapper around SMPLX body model class.
    from https://github.com/davrempe/humor/blob/main/humor/body_model/body_model.py
    '''

    def __init__(self,
                 bm_path,
                 num_betas=10,
                 batch_size=1,
                 num_expressions=100,
                 model_type='smplh',
                 regressor_path=None):
        super(BodyModel, self).__init__()
        '''
        Creates the body model object at the given path.

        :param bm_path: path to the body model pkl file
        :param num_expressions: only for smplx
        :param model_type: one of [smpl, smplh, smplx]
        :param use_vtx_selector: if true, returns additional vertices as joints that correspond to OpenPose joints
        '''

        kwargs = {
            'model_type': model_type,
            'num_betas': num_betas,
            'batch_size': batch_size,
            'num_expression_coeffs': num_expressions,
            'use_pca': False,
            'flat_hand_mean': False,
            'use_face_contour': True,
        }
        self.num_expressions = num_expressions
        assert (model_type in ['smpl', 'smplh', 'smplx'])
        if model_type == 'smpl':
            self.bm = SMPL(bm_path, **kwargs)
            self.num_joints = SMPL.NUM_JOINTS
        elif model_type == 'smplh':
            # smplx does not support .npz by default, so have to load in manually
            smpl_dict = np.load(bm_path, encoding='latin1')
            data_struct = Struct(**smpl_dict)
            # print(smpl_dict.files)
            if model_type == 'smplh':
                data_struct.hands_componentsl = np.zeros((0))
                data_struct.hands_componentsr = np.zeros((0))
                data_struct.hands_meanl = np.zeros((15 * 3))
                data_struct.hands_meanr = np.zeros((15 * 3))
                V, D, B = data_struct.shapedirs.shape
                data_struct.shapedirs = np.concatenate(
                    [data_struct.shapedirs, np.zeros((V, D, SMPL.SHAPE_SPACE_DIM - B))],
                    axis=-1)  # super hacky way to let smplh use 16-size beta
            kwargs['data_struct'] = data_struct
            self.bm = SMPLH(bm_path, **kwargs)
            self.num_joints = SMPLH.NUM_JOINTS
        elif model_type == 'smplx':
            self.bm = SMPLX(bm_path, **kwargs)
            self.left_hand_mean = self.bm.left_hand_mean
            self.right_hand_mean = self.bm.right_hand_mean
            self.hand_mean = torch.cat([self.left_hand_mean, self.right_hand_mean], dim=0)
            self.num_joints = SMPLX.NUM_JOINTS
            # create mean poses and shape for fitting initialization
            smpl_mean_params = np.load(constants.SMPL_MEAN_PATH)
            rot6d_poses = torch.tensor(smpl_mean_params['pose'], dtype=torch.float32)
            axis_poses = rot6d_to_axis_angle(rot6d_poses.reshape(-1, 6)).reshape(-1)
            mean_poses = self.bm.pose_mean.clone()
            mean_poses[:22 * 3] = axis_poses[:22 * 3]
            self.register_buffer('mean_poses', mean_poses)  # [165]
            self.register_buffer('mean_shape', torch.tensor(smpl_mean_params['shape'], dtype=torch.float32))  # [10]
            # misc data for evaluation
            self.face_vertex_idx = np.load(
                os.path.join(constants.BODY_MODEL_DIR, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))
            with open(os.path.join(constants.BODY_MODEL_DIR, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
                self.hand_vertex_idx = pickle.load(f, encoding='latin1')
            self.vertex_num = 10475
        self.model_type = model_type
        self.faces = self.bm.faces_tensor.numpy()

        if regressor_path is None and os.path.exists(regressor_paths[0]):
            regressor_path = regressor_paths
        if regressor_path is not None:
            for model_path in regressor_path:
                if 'SMPLX_to_J14.pkl' in model_path:
                    with open(model_path, 'rb') as f:
                        # Hand4Whole use it to evalute the EHF dataset
                        self.j14_regressor = pickle.load(f, encoding='latin1')
                elif 'J_regressor_h36m.npy' in model_path:
                    # Use it to evaluate GFPose trained on H36M
                    self.j17_regressor = np.load(model_path)

        self.J_regressor = self.bm.J_regressor.numpy()
        if model_type == 'smplx':
            self.orig_hand_regressor = self.make_hand_regressor()
        self.J_regressor_idx = {'pelvis': 0, 'lwrist': 20, 'rwrist': 21, 'neck': 12}
        self.openpose_mapping = smpl_to_openpose(model_type=model_type, use_face_contour=True)

    def make_hand_regressor(self, ):
        regressor = self.J_regressor.copy()
        lhand_regressor = np.concatenate((regressor[[20, 37, 38, 39], :], np.eye(self.vertex_num)[5361, None],
                                          regressor[[25, 26, 27], :], np.eye(self.vertex_num)[4933, None],
                                          regressor[[28, 29, 30], :], np.eye(self.vertex_num)[5058, None],
                                          regressor[[34, 35, 36], :], np.eye(self.vertex_num)[5169, None],
                                          regressor[[31, 32, 33], :], np.eye(self.vertex_num)[5286, None]))
        rhand_regressor = np.concatenate((regressor[[21, 52, 53, 54], :], np.eye(self.vertex_num)[8079, None],
                                          regressor[[40, 41, 42], :], np.eye(self.vertex_num)[7669, None],
                                          regressor[[43, 44, 45], :], np.eye(self.vertex_num)[7794, None],
                                          regressor[[49, 50, 51], :], np.eye(self.vertex_num)[7905, None],
                                          regressor[[46, 47, 48], :], np.eye(self.vertex_num)[8022, None]))
        hand_regressor = {'left': lhand_regressor, 'right': rhand_regressor}
        return hand_regressor

    def forward(self, global_orient=None, body_pose=None, left_hand_pose=None, right_hand_pose=None,
                jaw_pose=None, eye_poses=None, expression=None, betas=None, trans=None, dmpls=None,
                wholebody_params=None, return_dict=False, **kwargs):
        '''
        Note dmpls are not supported.
        '''
        assert (dmpls is None)
        assert 'pose_body' not in kwargs, 'use body_pose instead of pose_body'
        if wholebody_params is not None:  # [batchsize, 63+90+3+100], body, two_hands, jaw, expression
            assert self.model_type == 'smplx' and wholebody_params.shape[1] == 156 + self.num_expressions
            body_pose = wholebody_params[:, :63]
            left_hand_pose = wholebody_params[:, 63:63 + 45]
            right_hand_pose = wholebody_params[:, 63 + 45:63 + 45 + 45]
            jaw_pose = wholebody_params[:, 63 + 90:63 + 90 + 3]
            expression = wholebody_params[:, 63 + 90 + 3:]

        # parameters of SMPL should not be updated
        out_obj = self.bm(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=trans,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=None if eye_poses is None else eye_poses[:, :3],
            reye_pose=None if eye_poses is None else eye_poses[:, 3:],
            return_full_pose=True,
            **kwargs
        )

        out = {
            'v': out_obj.vertices,
            'f': self.bm.faces_tensor,
            'betas': out_obj.betas,
            'Jtr': out_obj.joints,
            'OpJtr': out_obj.joints[:, self.openpose_mapping],  # only openpose joints
            'body_joints': out_obj.joints[:, :22],  # only body joints
            'body_pose': out_obj.body_pose,
            'full_pose': out_obj.full_pose,
            'global_orient': out_obj.global_orient,
            'transl': trans,
        }
        if self.model_type in ['smplh', 'smplx']:
            out['hand_poses'] = torch.cat([out_obj.left_hand_pose, out_obj.right_hand_pose], dim=-1)
        if self.model_type == 'smplx':
            out['jaw_pose'] = out_obj.jaw_pose
            out['expression'] = out_obj.expression
            out['eye_poses'] = eye_poses

        if not return_dict:
            out = Struct(**out)

        return out
