import numpy as np
import torch

from lib.body_model.hand_model import MANO
from lib.body_model.utils import HandPartIndices, get_mano_skeleton
from lib.body_model.visual import visualize_3d_skeleton


class Evaler:
    def __init__(self, hand_model, part=None):
        self.hand_model = hand_model
        self.part = part

        if self.part is not None:
            # skip wrist. Since we don't model it, we shouldn't also evaluate this joint
            self.joint_idx = np.array(HandPartIndices.get_joint_indices(self.part))
        else:
            self.joint_idx = slice(None)

    def eval_hands(self, outs, gts):
        '''
        :param outs: [b, j*3] axis-angle results of hand poses
        :param gts:  [b, j*3] axis-angle groundtruth of hand poses
        :return: result dict for every sample [b,]
        '''
        eval_result = {'mpjpe': []}
        joint_gt_hand = self.hand_model(hand_pose=gts).Jtr[:, self.joint_idx]
        joint_out_hand = self.hand_model(hand_pose=outs).Jtr[:, self.joint_idx]
        mpjpe = torch.sqrt(torch.sum((joint_out_hand - joint_gt_hand) ** 2, dim=2)).mean(dim=1) * 1000
        eval_result['mpjpe'] = mpjpe.detach().cpu().numpy()

        return eval_result


    def multi_eval_hands(self, outs, gts):
        '''
        :param outs: [b, hypo, j*3] axis-angle results of hand poses, multiple hypothesis
        :param gts:  [b, j*3] axis-angle groundtruth of hand poses
        :return: result dict [b,]
        '''
        hypo_num = outs.shape[1]
        eval_result = {f'mpjpe': []}
        for hypo in range(hypo_num):
            result = self.eval_hands(outs[:, hypo], gts)
            eval_result['mpjpe'].append(result['mpjpe'])

        eval_result['mpjpe'] = np.min(eval_result['mpjpe'], axis=0)

        return eval_result

    def multi_eval_hands_all(self, outs, gts):
        '''
        :param outs: [b, hypo, j*3] axis-angle results of hand poses, multiple hypothesis
        :param gts:  [b, j*3] axis-angle groundtruth of hand poses
        :return: result dict [b,]
        '''
        hypo_num = outs.shape[1]
        eval_collector = {f'mpjpe': []}
        eval_result = {f'mpjpe_best': [], f'mpjpe_mean': [], f'mpjpe_std': []}
        for hypo in range(hypo_num):
            result = self.eval_hands(outs[:, hypo], gts)
            eval_collector['mpjpe'].append(result['mpjpe'])

        eval_result['mpjpe_best'] = np.min(eval_collector['mpjpe'], axis=0)
        eval_result['mpjpe_mean'] = np.mean(eval_collector['mpjpe'], axis=0)
        eval_result['mpjpe_std'] = np.std(eval_collector['mpjpe'], axis=0)

        return eval_result

    def print_eval_result(self, eval_result):
        print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))

    def print_multi_eval_result(self, eval_result, hypo_num):
        print(f'multihypo {hypo_num} MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))

    def print_multi_eval_result_all(self, eval_result, hypo_num):
        print(f'multihypo {hypo_num} MPJPE: %.2f mm' % np.mean(eval_result['mpjpe_mean']))
        print(f'multihypo {hypo_num} MPJPE std: %.2f mm' % np.mean(eval_result['mpjpe_std']))
        print(f'multihypo {hypo_num} MPJPE best: %.2f mm' % np.mean(eval_result['mpjpe_best']))


class IKEvaler:
    def __init__(self, hand_model, mask_idx=None):
        self.hand_model = hand_model
        # We don't have the part vertices index for the hand mesh, so only evaluate the joints if masked
        if mask_idx is not None:
            self.mask_idx = mask_idx
            self.visible_idx = [i for i in range(21) if i not in mask_idx]
            self.eval_keys = ['mpvpe', 'mpjpe_all', 'mpjpe_visible', 'mpjpe_masked']
        else:
            self.mask_idx = None
            self.eval_keys = ['mpvpe', 'mpjpe_all']

    def eval_hand(self, outs, gts):
        '''
        :param outs: [b, j*3] axis-angle results of hand poses
        :param gts:  [b, j*3] axis-angle groundtruth of hand poses
        :return: result dict for every sample [b,]
        '''
        eval_result = {}
        for key in self.eval_keys:
            eval_result[key] = []
        hand_gt = self.hand_model(hand_pose=gts)
        hand_out = self.hand_model(hand_pose=outs)

        # MPVPE from all vertices
        mesh_gt = hand_gt.v
        mesh_out = hand_out.v

        mpvpe = torch.sqrt(torch.sum((mesh_out - mesh_gt) ** 2, dim=2)).mean(dim=1) * 1000
        eval_result['mpvpe'] = mpvpe.detach().cpu().numpy()
        if self.mask_idx is not None:
            joint_gt_hand = hand_gt.OpJtr[:, self.mask_idx, :]
            joint_out_hand = hand_out.OpJtr[:, self.mask_idx, :]
            mpjpe = torch.sqrt(torch.sum((joint_out_hand - joint_gt_hand) ** 2, dim=2)).mean(dim=1) * 1000
            eval_result['mpjpe_masked'] = mpjpe.detach().cpu().numpy()
            joint_gt_hand = hand_gt.OpJtr[:, self.visible_idx, :]
            joint_out_hand = hand_out.OpJtr[:, self.visible_idx, :]
            mpjpe = torch.sqrt(torch.sum((joint_out_hand - joint_gt_hand) ** 2, dim=2)).mean(dim=1) * 1000
            eval_result['mpjpe_visible'] = mpjpe.detach().cpu().numpy()
            joint_gt_hand = hand_gt.OpJtr
            joint_out_hand = hand_out.OpJtr
            mpjpe = torch.sqrt(torch.sum((joint_out_hand - joint_gt_hand) ** 2, dim=2)).mean(dim=1) * 1000
            eval_result['mpjpe_all'] = mpjpe.detach().cpu().numpy()
        else:
            joint_gt_hand = hand_gt.Jtr
            joint_out_hand = hand_out.Jtr
            mpjpe = torch.sqrt(torch.sum((joint_out_hand - joint_gt_hand) ** 2, dim=2)).mean(dim=1) * 1000
            eval_result['mpjpe_all'] = mpjpe.detach().cpu().numpy()

        return eval_result

    def multi_eval_hand(self, outs, gts):
        '''
        :param outs: [b, hypo, j*3] axis-angle results of hand poses, multiple hypothesis
        :param gts:  [b, j*3] axis-angle groundtruth of hand poses
        :return: result dict [b,]
        '''
        hypo_num = outs.shape[1]
        eval_result = {}
        for key in self.eval_keys:
            eval_result[key] = []
        for hypo in range(hypo_num):
            result = self.eval_hand(outs[:, hypo], gts)
            for key in self.eval_keys:
                eval_result[key].append(result[key])

        for key in self.eval_keys:
            eval_result[key] = np.min(eval_result[key], axis=0)

        return eval_result

    def print_eval_result(self, eval_result):
        for key in self.eval_keys:
            print(f'{key}: %.2f mm' % np.mean(eval_result[key]))


if __name__ == '__main__':
    kpt_3d_vis = np.ones((16, 1))
    kps_lines = get_mano_skeleton()

    model = MANO(model_path='/data3/ljz24/projects/3d/body_models/mano', is_rhand=True,
                 batch_size=1)
    faces = model.faces
    import torch
    mano_params = {'betas': torch.zeros(1, 10, dtype=torch.float32),
                   'hand_pose': model.mean_pose,
                   }

    output = model(**mano_params)

    visualize_3d_skeleton(output.Jtr.detach().cpu().numpy()[0], kpt_3d_vis, kps_lines, output_path='skeleton.jpg')