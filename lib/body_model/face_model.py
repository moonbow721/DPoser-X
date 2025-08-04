import os

import torch
import numpy as np

import smplx
from smplx.utils import to_tensor, Struct


class FLAME(smplx.FLAME):
    def __init__(self,
                 model_path: str, num_expressions: int = 100, batch_size: int = 1,
                 use_face_contour: bool = True, num_betas: int = 10,
                 ):
        assert os.path.isdir(model_path)
        kwargs = {
            'model_path': model_path,
            'num_expression_coeffs': num_expressions,
            'num_betas': num_betas,
            'batch_size': batch_size,
            'use_face_contour': use_face_contour,
        }
        super(FLAME, self).__init__(**kwargs)
        face_mapping = np.arange(5+0, 5+51, dtype=np.int32)  # skip neck, backheads, eyeballs
        if use_face_contour:
            face_mapping = np.concatenate((np.arange(56, 56 + 17, dtype=np.int32), face_mapping))
        self.register_buffer('joint_map', torch.from_numpy(face_mapping).long())

    def forward(self, face_pose=None, face_params=None, full_face_params=None, *args, **kwargs):
        # "global_orient", "transl", "betas" can be passed as kwargs
        if face_pose is not None:   # [batchsize, 12], neck, jaw, leye, reye
            face_pose_dict = {'neck_pose': face_pose[:, :3], 'jaw_pose': face_pose[:, 3:6],
                              'leye_pose': face_pose[:, 6:9], 'reye_pose': face_pose[:, 9:12]}
            kwargs.update(face_pose_dict)
        if face_params is not None:     # [batchsize, 3+100], jaw, expression
            face_pose_dict = {'jaw_pose': face_params[:, :3], 'expression': face_params[:, 3:]}
            kwargs.update(face_pose_dict)
        if full_face_params is not None:    # [batchsize, 3+100+100], jaw, expression, betas
            face_pose_dict = {'jaw_pose': full_face_params[:, :3],
                              'expression': full_face_params[:, 3:self.num_expression_coeffs+3],
                              'betas': full_face_params[:, self.num_expression_coeffs+3:]
                              }
            kwargs.update(face_pose_dict)
        flame_output = super(FLAME, self).forward(*args, return_full_pose=True, **kwargs)

        out = {
            'v': flame_output.vertices,
            'f': self.faces_tensor,
            'betas': flame_output.betas,
            'expression': flame_output.expression,
            'Jtr': flame_output.joints,  # 56 Or 73 (+17 contour) joints
            'OpJtr': flame_output.joints[:, self.joint_map],  # 51 Or 68 (+17 contour) joints for Openpose
            'full_pose': flame_output.full_pose,
            'global_orient': flame_output.global_orient,
            'jaw_pose': flame_output.jaw_pose,
        }

        return Struct(**out)

