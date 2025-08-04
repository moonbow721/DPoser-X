import numpy as np
import torch


class IKEvaler:
    def __init__(self, face_model, mask_idx=None):
        self.face_model = face_model
        # We don't have the part vertices index for the face mesh, so only evaluate the joints if masked
        if mask_idx is not None:
            self.mask_idx = mask_idx
            self.visible_idx = [i for i in range(73) if i not in mask_idx]
            self.eval_keys = ['mpvpe', 'mpjpe_all', 'mpjpe_visible', 'mpjpe_masked']
        else:
            self.mask_idx = None
            self.eval_keys = ['mpvpe', 'mpjpe_all']

    def eval_face(self, outs, gts):
        '''
        :param outs: [b, j*3] axis-angle results of face poses
        :param gts:  [b, j*3] axis-angle groundtruth of face poses
        :return: result dict for every sample [b,]
        '''
        eval_result = {}
        for key in self.eval_keys:
            eval_result[key] = []
        face_gt = self.face_model(full_face_params=gts)
        face_out = self.face_model(full_face_params=outs)

        # MPVPE from all vertices
        mesh_gt = face_gt.v
        mesh_out = face_out.v

        mpvpe = torch.sqrt(torch.sum((mesh_out - mesh_gt) ** 2, dim=2)).mean(dim=1) * 1000
        eval_result['mpvpe'] = mpvpe.detach().cpu().numpy()
        if self.mask_idx is not None:
            joint_gt_face = face_gt.Jtr[:, self.mask_idx, :]
            joint_out_face = face_out.Jtr[:, self.mask_idx, :]
            mpjpe = torch.sqrt(torch.sum((joint_out_face - joint_gt_face) ** 2, dim=2)).mean(dim=1) * 1000
            eval_result['mpjpe_masked'] = mpjpe.detach().cpu().numpy()
            joint_gt_face = face_gt.Jtr[:, self.visible_idx, :]
            joint_out_face = face_out.Jtr[:, self.visible_idx, :]
            mpjpe = torch.sqrt(torch.sum((joint_out_face - joint_gt_face) ** 2, dim=2)).mean(dim=1) * 1000
            eval_result['mpjpe_visible'] = mpjpe.detach().cpu().numpy()
            joint_gt_face = face_gt.Jtr
            joint_out_face = face_out.Jtr
            mpjpe = torch.sqrt(torch.sum((joint_out_face - joint_gt_face) ** 2, dim=2)).mean(dim=1) * 1000
            eval_result['mpjpe_all'] = mpjpe.detach().cpu().numpy()
        else:
            joint_gt_face = face_gt.Jtr
            joint_out_face = face_out.Jtr
            mpjpe = torch.sqrt(torch.sum((joint_out_face - joint_gt_face) ** 2, dim=2)).mean(dim=1) * 1000
            eval_result['mpjpe_all'] = mpjpe.detach().cpu().numpy()

        return eval_result

    def multi_eval_face(self, outs, gts):
        '''
        :param outs: [b, hypo, j*3] axis-angle results of face poses, multiple hypothesis
        :param gts:  [b, j*3] axis-angle groundtruth of face poses
        :return: result dict [b,]
        '''
        hypo_num = outs.shape[1]
        eval_result = {}
        for key in self.eval_keys:
            eval_result[key] = []
        for hypo in range(hypo_num):
            result = self.eval_face(outs[:, hypo], gts)
            for key in self.eval_keys:
                eval_result[key].append(result[key])

        for key in self.eval_keys:
            eval_result[key] = np.min(eval_result[key], axis=0)

        return eval_result

    def print_eval_result(self, eval_result):
        for key in self.eval_keys:
            print(f'{key}: %.2f mm' % np.mean(eval_result[key]))

