import os
import pickle

import numpy as np
import torch

from lib.body_model import constants
from lib.body_model.utils import OpWholeBodyPartIndices


class Evaler:
    def __init__(self, body_model, part=None):
        self.body_model = body_model
        self.part = part
        assert part in ['body', 'lhand', 'rhand', 'face', None]

        with open(os.path.join(constants.BODY_MODEL_DIR, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            self.hand_vertex_idx = pickle.load(f, encoding='latin1')

        if self.part is not None:
            self.joint_idx = np.array(OpWholeBodyPartIndices.get_joint_indices(self.part))
        else:
            self.joint_idx = slice(None)

        if self.part == 'lhand':
            self.vertex_idx = self.hand_vertex_idx['left_hand']
        elif self.part == 'rhand':
            self.vertex_idx = self.hand_vertex_idx['right_hand']
        else:
            self.vertex_idx = slice(None)

    def eval_bodys(self, outs, gts):
        '''
        :param outs: [b, j*3] axis-angle results of body poses
        :param gts:  [b, j*3] axis-angle groundtruth of body poses
        :return: result dict for every sample [b,]
        '''
        eval_result = {}
        gt_body = self.body_model(wholebody_params=gts)
        out_body = self.body_model(wholebody_params=outs)
        joint_gt_part = gt_body.Jtr[:, self.joint_idx]
        joint_out_part = out_body.Jtr[:, self.joint_idx]
        mpjpe = torch.sqrt(torch.sum((joint_out_part - joint_gt_part) ** 2, dim=2)).mean(dim=1) * 1000
        eval_result['mpjpe'] = mpjpe.detach().cpu().numpy()
        vert_gt_body = gt_body.v
        vert_out_body = out_body.v
        vert_gt_part = vert_gt_body[:, self.vertex_idx]
        vert_out_part = vert_out_body[:, self.vertex_idx]
        mpvpe = torch.sqrt(torch.sum((vert_out_part - vert_gt_part) ** 2, dim=2)).mean(dim=1) * 1000
        eval_result['mpvpe'] = mpvpe.detach().cpu().numpy()

        return eval_result

    def multi_eval_bodys(self, outs, gts):
        '''
        :param outs: [b, hypo, j*3] axis-angle results of body poses, multiple hypothesis
        :param gts:  [b, j*3] axis-angle groundtruth of body poses
        :return: result dict [b,]
        '''
        hypo_num = outs.shape[1]
        eval_result = {'mpjpe': [], 'mpvpe': []}
        for hypo in range(hypo_num):
            result = self.eval_bodys(outs[:, hypo], gts)
            eval_result['mpjpe'].append(result['mpjpe'])
            eval_result['mpvpe'].append(result['mpvpe'])

        eval_result['mpjpe'] = np.min(eval_result['mpjpe'], axis=0)
        eval_result['mpvpe'] = np.min(eval_result['mpvpe'], axis=0)

        return eval_result

    def multi_eval_bodys_all(self, outs, gts):
        '''
        :param outs: [b, hypo, j*3] axis-angle results of body poses, multiple hypothesis
        :param gts:  [b, j*3] axis-angle groundtruth of body poses
        :return: result dict [b,]
        '''
        hypo_num = outs.shape[1]
        eval_collector = {f'mpjpe': [], f'mpvpe': []}
        eval_result = {f'mpjpe_best': [], f'mpjpe_mean': [], f'mpjpe_std': [],
                       f'mpvpe_best': [], f'mpvpe_mean': [], f'mpvpe_std': []}
        for hypo in range(hypo_num):
            result = self.eval_bodys(outs[:, hypo], gts)
            eval_collector['mpjpe'].append(result['mpjpe'])
            eval_collector['mpvpe'].append(result['mpvpe'])

        eval_result['mpjpe_best'] = np.min(eval_collector['mpjpe'], axis=0)
        eval_result['mpjpe_mean'] = np.mean(eval_collector['mpjpe'], axis=0)
        eval_result['mpjpe_std'] = np.std(eval_collector['mpjpe'], axis=0)
        eval_result['mpvpe_best'] = np.min(eval_collector['mpvpe'], axis=0)
        eval_result['mpvpe_mean'] = np.mean(eval_collector['mpvpe'], axis=0)
        eval_result['mpvpe_std'] = np.std(eval_collector['mpvpe'], axis=0)

        return eval_result

    def print_eval_result(self, eval_result):
        print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
        print('MPVPE: %.2f mm' % np.mean(eval_result['mpvpe']))

    def print_multi_eval_result(self, eval_result, hypo_num):
        print(f'multihypo {hypo_num} MPJPE best: %.2f mm' % np.mean(eval_result['mpjpe']))
        print(f'multihypo {hypo_num} MPVPE best: %.2f mm' % np.mean(eval_result['mpvpe']))

    def print_multi_eval_result_all(self, eval_result, hypo_num):
        print(f'multihypo {hypo_num} MPJPE mean: %.2f mm' % np.mean(eval_result['mpjpe_mean']))
        print(f'multihypo {hypo_num} MPJPE std: %.2f mm' % np.mean(eval_result['mpjpe_std']))
        print(f'multihypo {hypo_num} MPJPE best: %.2f mm' % np.mean(eval_result['mpjpe_best']))
        print(f'multihypo {hypo_num} MPVPE mean: %.2f mm' % np.mean(eval_result['mpvpe_mean']))
        print(f'multihypo {hypo_num} MPVPE std: %.2f mm' % np.mean(eval_result['mpvpe_std']))
        print(f'multihypo {hypo_num} MPVPE best: %.2f mm' % np.mean(eval_result['mpvpe_best']))
