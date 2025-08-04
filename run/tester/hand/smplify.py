import os
import pickle

import cv2
import numpy as np
import torch
from torch import nn

from lib.algorithms.advanced import sde_lib
from lib.algorithms.advanced import utils as mutils
from lib.algorithms.advanced.model import create_model
from lib.body_model.fitting_losses import camera_fitting_loss, body_fitting_loss
from lib.dataset.hand import N_POSES
from lib.dataset.utils import Posenormalizer
from lib.utils.generic import import_configs, load_model
from lib.utils.misc import lerp


class DPoser(nn.Module):
    def __init__(self, batch_size=32, config_path='', args=None):
        super().__init__()
        self.device = args.device
        self.batch_size = batch_size
        config = import_configs(config_path)

        self.Normalizer = Posenormalizer(
            data_path=os.path.join(args.data_path, 'hand_normalizer'),
            normalize=config.data.normalize, min_max=config.data.min_max, rot_rep=config.data.rot_rep,
            device=args.device)

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
        POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
        model = create_model(config.model, N_POSES, POSE_DIM)
        model.to(self.device)
        model.eval()
        load_model(model, config.model, args.ckpt_path, args.device, is_ema=True)
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
        # x_0: [B, j*6], vec_t: [B], quan_t: [1]
        z = torch.randn_like(x_0)
        mean, std = self.sde.marginal_prob(x_0, vec_t)
        perturbed_data = mean + std[:, None] * z  #
        if multi_denoise:
            denoise_data, SNR = self.multi_step_denoise(perturbed_data, vec_t, t_end=vec_t / (2 * 5), N=5)
        else:
            denoise_data, SNR = self.one_step_denoise(perturbed_data, vec_t)
        weight = 0.5 * torch.sqrt(1 + SNR**2)
        loss = torch.sum(weight * self.loss_fn(x_0, denoise_data)) / self.batch_size

        return loss

    def forward(self, poses, betas, t):
        poses = self.Normalizer.offline_normalize(poses[:, :N_POSES * 3], from_axis=True)
        vec_t = torch.ones(self.batch_size, device=self.device) * t
        prior_loss = self.DPoser_loss(poses, vec_t)
        return prior_loss


class SMPLify:
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 body_model,
                 step_size=1e-2,
                 batch_size=32,
                 num_iters=100,
                 focal_length=5000,
                 args=None):
        self.smpl = body_model
        self.batch_size = batch_size
        # Store options
        self.args = args
        self.device = args.device
        self.focal_length = focal_length
        self.step_size = step_size

        self.num_iters = num_iters
        self.prior_name = args.prior

        self.loss_weights = {'pose_prior_weight': [30, 10, 10, 5, 5],
                             'shape_prior_weight': [50, 20, 10, 5, 5],
                             'angle_prior_weight': [0, 0, 0, 0, 0],
                             }
        self.stages = len(self.loss_weights['pose_prior_weight'])
        self.pelvis_conf_weight = 2.0

        self.time_strategy = self.args.time_strategy
        self.t_max = 0.12
        self.t_min = 0.08
        self.fixed_t = 0.1

    def load_prior(self, prior=None):
        if prior is not None:
            self.pose_prior = prior
        else:
            self.pose_prior = None

    def sample_continuous_time(self, iteration):
        total_steps = self.stages * self.num_iters
        if self.prior_name == 'DPoser':
            if self.time_strategy == '1':
                t = self.pose_prior.eps + torch.rand(1, device=self.device) * (self.pose_prior.sde.T - self.pose_prior.eps)
            elif self.time_strategy == '2':
                t = torch.tensor(self.fixed_t)
            elif self.time_strategy == '3':
                t = self.t_min + torch.tensor(total_steps - iteration - 1) / total_steps * (self.t_max - self.t_min)
            else:
                raise NotImplementedError
        else:
            t = 0

        return t

    def __call__(self, init_pose, init_betas, init_cam_t, camera_center, keypoints_2d):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """
        batch_size = init_pose.shape[0]

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2].clone()
        joints_conf = keypoints_2d[:, :, -1].clone()
        joints_conf[:, 0] = joints_conf[:, 0] * self.pelvis_conf_weight

        # Split SMPL pose to body pose and global orientation
        hand_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 0: Optimize camera translation
        hand_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = False
        camera_translation.requires_grad = True
        camera_opt_params = [
            {'params': [camera_translation], 'lr': 0.1}
        ]
        camera_optimizer = torch.optim.Adam(camera_opt_params)
        for i in range(50):
            smpl_output = self.smpl(betas=betas,
                                    hand_pose=hand_pose,
                                    global_orient=global_orient,
                                    transl=camera_translation)

            model_joints = smpl_output.OpJtr
            loss = camera_fitting_loss(model_joints, camera_translation, init_cam_t, camera_center,
                                       joints_2d, joints_conf, focal_length=self.focal_length, part='rhand')

            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        # Step 1: Optimize camera translation and body orientation
        init_cam_t = camera_translation.clone()
        global_orient.requires_grad = True
        camera_translation.requires_grad = True
        camera_opt_params = [
            {'params': [global_orient], 'lr': 0.05},
            {'params': [camera_translation], 'lr': 0.01}
        ]
        camera_optimizer = torch.optim.Adam(camera_opt_params)

        for i in range(100):
            smpl_output = self.smpl(betas=betas,
                                    hand_pose=hand_pose,
                                    global_orient=global_orient,
                                    transl=camera_translation)

            model_joints = smpl_output.OpJtr
            loss = camera_fitting_loss(model_joints, camera_translation, init_cam_t, camera_center,
                                       joints_2d, joints_conf, focal_length=self.focal_length, part='rhand',)

            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        hand_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [
            {'params': [hand_pose], 'lr': self.step_size},
            {'params': [betas], 'lr': self.step_size},
            {'params': [global_orient], 'lr': 0.05}
        ]

        body_optimizer = torch.optim.Adam(body_opt_params, betas=(0.9, 0.999))

        stage_weights = [dict(zip(self.loss_weights.keys(), vals)) for vals in zip(*self.loss_weights.values())]

        for stage, current_weights in enumerate(stage_weights[:-1]):
            for i in range(self.num_iters):
                smpl_output = self.smpl(betas=betas,
                                        hand_pose=hand_pose,
                                        global_orient=global_orient,
                                        transl=camera_translation)

                model_joints = smpl_output.OpJtr
                t = self.sample_continuous_time(iteration=stage * self.num_iters + i)

                loss = body_fitting_loss(hand_pose, betas, model_joints, camera_translation, camera_center,
                                         joints_2d, joints_conf, self.pose_prior, t=t, part='rhand',
                                         focal_length=self.focal_length, **current_weights, verbose=False)

                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Step 3: Optimize all parameters
        hand_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True
        body_opt_params = [
            {'params': [hand_pose], 'lr': self.step_size},
            {'params': [betas], 'lr': self.step_size},
            {'params': [global_orient], 'lr': 0.01},
            {'params': [camera_translation], 'lr': 0.01}
        ]

        body_optimizer = torch.optim.Adam(body_opt_params, betas=(0.9, 0.999))

        stage = len(stage_weights) - 1
        for i in range(self.num_iters):
            smpl_output = self.smpl(betas=betas,
                                    hand_pose=hand_pose,
                                    global_orient=global_orient,
                                    transl=camera_translation)

            model_joints = smpl_output.OpJtr
            t = self.sample_continuous_time(iteration=stage * self.num_iters + i)

            loss = body_fitting_loss(hand_pose, betas, model_joints, camera_translation, camera_center,
                                     joints_2d, joints_conf, self.pose_prior, t=t, part='rhand',
                                     focal_length=self.focal_length, **stage_weights[-1], verbose=False)

            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

        # Get final loss value
        with torch.no_grad():

            smpl_output = self.smpl(betas=betas,
                                    hand_pose=hand_pose,
                                    global_orient=global_orient,
                                    transl=camera_translation)

            model_joints = smpl_output.OpJtr
            t = self.sample_continuous_time(iteration=stage * self.num_iters + i)
            reprojection_loss = body_fitting_loss(hand_pose, betas, model_joints, camera_translation, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior, t=t, part='rhand',
                                                  focal_length=self.focal_length, output='reprojection', verbose=False)

        pose = torch.cat([global_orient, hand_pose], dim=-1).detach()
        betas = betas.detach()

        return pose, betas, camera_translation, reprojection_loss
