import sys
import os

import torch
import torch.nn as nn

from lib.algorithms.advanced import sde_lib
from lib.algorithms.advanced import utils as mutils
from lib.algorithms.advanced.model import create_model
from lib.dataset.face import N_POSES
from lib.dataset.utils import Posenormalizer, CombinedNormalizer
from lib.utils.generic import import_configs, load_model
from lib.utils.misc import lerp


class L2Prior(nn.Module):
    def __init__(self, *args, **kwargs):
        super(L2Prior, self).__init__()

    def forward(self, module_input, *args):
        return torch.mean(module_input.pow(2).sum(dim=-1))


class DPoser_pose_exp(nn.Module):
    def __init__(self, device, config_path='', args=None):
        super().__init__()
        self.device = device
        config = import_configs(config_path)

        data_path = {"jaw_pose": os.path.join(args.data_root, 'jaw_normalizer'),
                     "expression": os.path.join(args.data_root, 'expression_normalizer')}
        self.Normalizer = CombinedNormalizer(
            data_path_dict=data_path, model='face',
            normalize=config.data.normalize, min_max=config.data.min_max, rot_rep=config.data.rot_rep,
            device=device)

        diffusion_model = self.load_model(config, args)
        if config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                N=config.model.num_scales)
        elif config.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                   N=config.model.num_scales)
        elif config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                                N=config.model.num_scales)
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

        self.sde = sde
        self.eps = 1e-3
        self.score_fn = mutils.get_score_fn(sde, diffusion_model, train=False, continuous=config.training.continuous)
        self.rsde = sde.reverse(self.score_fn, False)
        # L2 loss
        self.loss_fn = nn.MSELoss(reduction='none')
        self.timesteps = torch.linspace(self.sde.T, self.eps, self.sde.N, device=self.device)

    def load_model(self, config, args):
        POSE_DIM = 1  # multiply 3 (jaw) + 100 (expression)
        POSES = N_POSES
        if config.data.rot_rep == 'rot6d':
            POSES += 3
        model = create_model(config.model, POSES, POSE_DIM)
        model.to(self.device)
        model.eval()
        load_model(model, config.model, args.exp_ckpt_path, self.device, is_ema=True)
        return model

    def one_step_denoise(self, x_t, t):
        drift, diffusion, alpha, sigma_2, score = self.rsde.sde(x_t, t, guide=True)
        x_0_hat = (x_t + sigma_2[:, None] * score) / alpha
        SNR = alpha / torch.sqrt(sigma_2)[:, None]

        return x_0_hat.detach(), SNR

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
            x_current = (alpha_before / alpha_current * (x_current - sigma_current[:, None] * score) +
                         sigma_before[:, None] * score)
        alpha, sigma = self.sde.return_alpha_sigma(time_traj[0])
        SNR = alpha / sigma[:, None]
        return x_current.detach(), SNR

    def DPoser_loss(self, x_0, vec_t, multi_denoise=False):
        batch_size = x_0.shape[0]
        z = torch.randn_like(x_0)
        mean, std = self.sde.marginal_prob(x_0, vec_t)
        perturbed_data = mean + std[:, None] * z  #
        if multi_denoise:
            denoise_data, SNR = self.multi_step_denoise(perturbed_data, vec_t, t_end=vec_t / (2 * 5), N=5)
        else:
            denoise_data, SNR = self.one_step_denoise(perturbed_data, vec_t)
        weight = 0.5 * torch.sqrt(1 + SNR**2)
        loss = torch.sum(weight * self.loss_fn(x_0, denoise_data)) / batch_size

        return loss

    def forward(self, params, t):    # params: [B, 3+100], jaw pose first
        batch_size = params.shape[0]
        params = self.Normalizer.offline_normalize(params, from_axis=True)
        vec_t = torch.ones(batch_size, device=self.device) * t
        prior_loss = self.DPoser_loss(params, vec_t)
        return prior_loss


class DPoser_beta(nn.Module):
    def __init__(self, device, config_path='', args=None):
        super().__init__()
        self.device = device
        config = import_configs(config_path)
        if '300' in config_path:
            data_path = os.path.join(args.data_root, 'betas300_normalizer')
            pose_num = 300
        else:
            data_path = os.path.join(args.data_root, 'betas_normalizer')
            pose_num = 100
        self.Normalizer = Posenormalizer(
            data_path=data_path, normalize=config.data.normalize,
            min_max=config.data.min_max, rot_rep=config.data.rot_rep,
            device=device)

        diffusion_model = self.load_model(config, args, pose_num)
        if config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                N=config.model.num_scales)
        elif config.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                   N=config.model.num_scales)
        elif config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                                N=config.model.num_scales)
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

        self.sde = sde
        self.eps = 1e-3
        self.score_fn = mutils.get_score_fn(sde, diffusion_model, train=False, continuous=config.training.continuous)
        self.rsde = sde.reverse(self.score_fn, False)
        # L2 loss
        self.loss_fn = nn.MSELoss(reduction='none')
        self.timesteps = torch.linspace(self.sde.T, self.eps, self.sde.N, device=self.device)

    def load_model(self, config, args, pose_num=100):
        POSE_DIM = 1
        model = create_model(config.model, pose_num, POSE_DIM)
        model.to(self.device)
        model.eval()
        load_model(model, config.model, args.beta_ckpt_path, self.device, is_ema=True)
        return model

    def one_step_denoise(self, x_t, t):
        drift, diffusion, alpha, sigma_2, score = self.rsde.sde(x_t, t, guide=True)
        x_0_hat = (x_t + sigma_2[:, None] * score) / alpha
        SNR = alpha / torch.sqrt(sigma_2)[:, None]

        return x_0_hat.detach(), SNR

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
            x_current = (alpha_before / alpha_current * (x_current - sigma_current[:, None] * score) +
                         sigma_before[:, None] * score)
        alpha, sigma = self.sde.return_alpha_sigma(time_traj[0])
        SNR = alpha / sigma[:, None]
        return x_current.detach(), SNR

    def DPoser_loss(self, x_0, vec_t, multi_denoise=False):
        batch_size = x_0.shape[0]
        z = torch.randn_like(x_0)
        mean, std = self.sde.marginal_prob(x_0, vec_t)
        perturbed_data = mean + std[:, None] * z  #
        if multi_denoise:
            denoise_data, SNR = self.multi_step_denoise(perturbed_data, vec_t, t_end=vec_t / (2 * 5), N=5)
        else:
            denoise_data, SNR = self.one_step_denoise(perturbed_data, vec_t)
        weight = 0.5 * torch.sqrt(1 + SNR**2)
        loss = torch.sum(weight * self.loss_fn(x_0, denoise_data)) / batch_size

        return loss

    def forward(self, params, t):    # params: [B, 100]
        batch_size = params.shape[0]
        params = self.Normalizer.offline_normalize(params, from_axis=True)
        vec_t = torch.ones(batch_size, device=self.device) * t
        prior_loss = self.DPoser_loss(params, vec_t)
        return prior_loss