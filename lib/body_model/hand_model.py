import os
import pickle
from typing import Optional

import numpy as np
import smplx
import torch
from smplx.lbs import vertices2joints
from smplx.utils import to_tensor, Struct
from smplx.vertex_ids import vertex_ids


# adapted from HaMaR
class MANO(smplx.MANO):
    def __init__(self,
                 model_path: str, is_rhand: bool = True, batch_size: int = 1,
                 joint_regressor_extra: Optional[str] = None,
                 num_betas: int = 10,
                 ):
        """
        Extension of the official MANO implementation to support more joints.
        Args:
            Same as MANOLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        assert os.path.isdir(model_path)
        kwargs = {
            'model_path': model_path,
            'is_rhand': is_rhand,
            'num_betas': num_betas,
            'batch_size': batch_size,
            'use_pca': False,
            'flat_hand_mean': False
        }
        super(MANO, self).__init__(**kwargs)
        mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        if not is_rhand:
            print('Fix shapedirs bug of left hand MANO [https://github.com/vchoutas/smplx/issues/48]')
            self.shapedirs[:, 0, :] *= -1

        # 2, 3, 5, 4, 1
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra',
                                 torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'),
                                              dtype=torch.float32))
        self.register_buffer('extra_joints_idxs', to_tensor(list(vertex_ids['mano'].values()), dtype=torch.long))
        self.register_buffer('joint_map', torch.tensor(mano_to_openpose, dtype=torch.long))

        mean_params = torch.load(os.path.join(model_path, 'mano_mean_params.pt'))
        self.register_buffer('mean_orient', mean_params['orient'].float().unsqueeze(0))
        self.register_buffer('mean_pose', mean_params['hand_pose'].float().unsqueeze(0))
        self.register_buffer('mean_poses', torch.cat([self.mean_orient, self.mean_pose], dim=1))
        self.register_buffer('mean_shape', mean_params['betas'].float().unsqueeze(0))

    def forward(self, *args, **kwargs):
        """
        Run forward pass. Same as MANO and also append an extra set of joints if joint_regressor_extra is specified.
        """
        mano_output = super(MANO, self).forward(*args, return_full_pose=True, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, mano_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)

        out = {
            'v': mano_output.vertices,
            'f': self.faces_tensor,
            'betas': mano_output.betas,
            'Jtr': mano_output.joints,  # 16 joints follow smplx order
            'OpJtr': joints,  # 21 joints follow OpenPose order
            'hand_pose': mano_output.hand_pose,
            'full_pose': mano_output.full_pose
        }

        return Struct(**out)


# flip left hand to right hand (MANO parmas)
def flip_hand(params):
    assert len(params.shape) == 2
    if isinstance(params, np.ndarray):
        fliped_params = params.copy()
    elif isinstance(params, torch.Tensor):
        fliped_params = params.clone()
    else:
        raise ValueError('params should be np.ndarray or torch.Tensor')
    fliped_params[:, 1::3] *= -1
    fliped_params[:, 2::3] *= -1
    return fliped_params

