import functools

import torch
import torch.nn as nn

from lib.algorithms.advanced.model import create_model
from lib.utils.generic import load_model, import_configs


def create_fullface_model(model_config, POSES_LIST, POSE_DIM):
    assert POSE_DIM == 1
    if model_config.type == 'Combiner':
        model = Combine_face_model(
            import_configs(model_config.pose_config).model,
            import_configs(model_config.shape_config).model,
            poses_list=POSES_LIST,
            pose_ckpt=model_config.pose_ckpt,
            shape_ckpt=model_config.shape_ckpt,
        )
    else:
        raise NotImplementedError('unsupported model')

    return model


class Combine_face_model(nn.Module):
    def __init__(self, pose_config, shape_config, poses_list, pose_ckpt, shape_ckpt,):
        super(Combine_face_model, self).__init__()
        self.pose_coefficients = poses_list[0]
        self.shape_coefficients = poses_list[1]

        self.pose_model = create_model(pose_config, poses_list[0], 1)
        self.shape_model = create_model(shape_config, poses_list[1], 1)
        load_model(self.pose_model, pose_config, pose_ckpt, 'cpu', is_ema=True)
        load_model(self.shape_model, shape_config, shape_ckpt, 'cpu', is_ema=True)

    def forward(self, batch, t, condition_list=None, mask_list=None):
        """
        batch: [B, j*3] or [B, j*6], Order: [jaw+exp, betas]
        t: [B]
        condition: not be enabled
        mask: [B, j*3] or [B, j*6] same dim as batch
        Return: [B, j*3] or [B, j*6] same dim as batch
        """
        if condition_list is None:
            condition_list = [None, None]
        if mask_list is None:
            mask_list = [None, None]
        jaw_exp, betas = (
            torch.split(batch, [self.pose_coefficients, self.shape_coefficients], dim=1))
        output_jaw_exp = self.pose_model(jaw_exp, t, condition_list[0], mask_list[0])
        output_betas = self.shape_model(betas, t, condition_list[1], mask_list[1])

        return torch.cat([output_jaw_exp, output_betas], dim=1)