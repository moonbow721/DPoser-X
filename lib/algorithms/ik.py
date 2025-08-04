import math

import numpy as np
import torch
import torch.nn as nn

from lib.utils.misc import lerp
from lib.algorithms.advanced import utils as mutils


class DPoserIK(object):
    def __init__(self, diffusion_model, body_model, normalize_fn, sde, continuous):
        self.body_model = body_model
        self.normalize_fn = normalize_fn
        self.sde = sde
        self.score_fn = mutils.get_score_fn(sde, diffusion_model, train=False, continuous=continuous)
        self.rsde = sde.reverse(self.score_fn, True)
        # L2 loss
        self.loss_fn = nn.MSELoss(reduction='none')
        self.data_loss = nn.MSELoss(reduction='mean')

    @torch.no_grad()
    def one_step_denoise(self, x_t, t):
        drift, diffusion, alpha, sigma_2, score = self.rsde.sde(x_t, t, guide=True)
        x_0_hat = (x_t + sigma_2[:, None] * score) / alpha

        return x_0_hat.detach()

    @torch.no_grad()
    def multi_step_denoise(self, x_t, t, t_end, N=10):
        time_traj = lerp(t, t_end, N + 1)
        x_current = x_t

        for i in range(N):
            t_current = time_traj[i]
            t_before = time_traj[i + 1]
            alpha_current, sigma_current = self.sde.return_alpha_sigma(t_current)
            alpha_before, sigma_before = self.sde.return_alpha_sigma(t_before)
            score = self.score_fn(x_current, t_current, condition=None, mask=None)
            score = -score * sigma_current[:, None]  # score to noise prediction
            x_current = (alpha_before / alpha_current * (x_current - sigma_current[:, None] * score)
                         + sigma_before[:, None] * score)

        return x_current.detach()

    def loss(self, x_0, vec_t, weighted=False, multi_denoise=False):
        # x_0: [B, j*6], vec_t: [B],
        z = torch.randn_like(x_0)
        mean, std = self.sde.marginal_prob(x_0, vec_t)
        perturbed_data = mean + std[:, None] * z  #
        alpha, sigma = self.sde.return_alpha_sigma(vec_t)
        SNR = alpha / sigma[:, None]
        if multi_denoise:  # not recommended
            t_end = torch.ones_like(vec_t) * 0.001
            denoise_data = self.multi_step_denoise(perturbed_data, vec_t, t_end=t_end, N=5)
        else:
            denoise_data = self.one_step_denoise(perturbed_data, vec_t)

        if weighted:
            weight = 0.5 * torch.sqrt(1 + SNR)
        else:
            weight = 0.5

        dposer_loss = torch.mean(weight * self.loss_fn(x_0, denoise_data))

        return dposer_loss

    def get_loss_weights(self, input_type='partial'):
        """Set loss weights"""
        if input_type == 'partial':
            loss_weight = {'data': lambda cst, it: 100 * cst * (1 + it),
                           'dposer': lambda cst, it: 0.1 * cst / (1 + it)}
        elif input_type == 'noisy':
            loss_weight = {'data': lambda cst, it: 10.0 * cst,
                           'dposer': lambda cst, it: 0.1 * cst}
        else:
            raise NotImplementedError
        return loss_weight

    @staticmethod
    def backward_step(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    def optimize(self, joints_3d, joints_mask, pose_init, time_strategy='3', lr=0.05,
                 t_max=0.15, t_min=0.05, t_fixed=0.1,
                 iterations=10, steps_per_iter=30, input_type='partial', prior_weight=1.0):
        batch_size = joints_3d.shape[0]
        joints_3d = joints_3d.detach()
        total_steps = iterations * steps_per_iter
        weight_dict = self.get_loss_weights(input_type)
        loss_dict = dict()
        with torch.enable_grad():
            opti_variable = pose_init.clone().detach()
            opti_variable.requires_grad = True
            optimizer = torch.optim.Adam([opti_variable], lr, betas=(0.9, 0.999))

            eps = 1e-3
            for it in range(iterations):
                for i in range(steps_per_iter):
                    step = it * steps_per_iter + i
                    optimizer.zero_grad()

                    '''   *************      DPoser loss ***********         '''
                    if time_strategy == '1':
                        t = eps + torch.rand(1, device=joints_3d.device) * (self.sde.T - eps)
                    elif time_strategy == '2':
                        t = torch.tensor(t_fixed)
                    elif time_strategy == '3':
                        t = t_min + torch.tensor(total_steps - step - 1) / total_steps * (t_max - t_min)
                    else:
                        raise NotImplementedError
                    vec_t = torch.ones(batch_size, device=joints_3d.device) * t
                    poses = self.normalize_fn(opti_variable, from_axis=True)
                    loss_dict['dposer'] = self.loss(poses, vec_t, ) * prior_weight
                    '''   ***********      DPoser loss   ************       '''
                    opti_joints = self.body_model(hand_pose=opti_variable).OpJtr
                    loss_dict['data'] = self.data_loss(opti_joints * joints_mask, joints_3d * joints_mask)

                    # Get total loss for backward pass
                    tot_loss = self.backward_step(loss_dict, weight_dict, it)
                    tot_loss.backward()
                    optimizer.step()

        return opti_variable
