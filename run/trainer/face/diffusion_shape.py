import argparse
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim

from lib.algorithms.advanced import losses, sde_lib, sampling, likelihood
from lib.algorithms.advanced.model import create_model
from lib.algorithms.ema import ExponentialMovingAverage
from lib.body_model.face_model import FLAME
from lib.body_model.visual import render_mesh
from lib.utils.callbacks import TimerCallback, ModelSizeCallback
from lib.utils.generic import import_configs
from lib.utils.metric import average_pairwise_distance, self_intersections_percentage
from lib.utils.schedulers import CosineWarmupScheduler
from lib.dataset.face import FlameDataModule
from lib.dataset.utils import Posenormalizer


def parse_args(argv):
    parser = argparse.ArgumentParser(description='train diffusion model of face shape')
    parser.add_argument('--config-path', '-c', type=str,
                        default='configs.face.subvp.shape_timefc.get_config',
                        help='config files to build DPoser')
    parser.add_argument('--bodymodel-path', type=str,
                        default='../body_models/flame',
                        help='load SMPLX for visualization')
    parser.add_argument('--resume-ckpt', '-r', type=str, help='resume training')
    parser.add_argument('--data-root', type=str,
                        default='./face_data', help='dataset root')

    parser.add_argument('--name', type=str, default='default', help='name of checkpoint folder')

    args = parser.parse_args(argv[1:])

    return args


class DPoserTrainer(pl.LightningModule):
    def __init__(self, config,
                 bodymodel_path='',
                 data_path='',
                 N_POSES=103,
                 train_loader=None,
                 val_loader=None, ):
        super().__init__()
        self.config = config
        self.bodymodel_path = bodymodel_path
        self.data_path = data_path
        self.N_POSES = N_POSES
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_hyperparameters(ignore=['train_loader', 'val_loader'])

        # Collect data
        self.last_trajs = None
        self.all_samples = []

        # Initialize the model
        self.POSE_DIM = 1
        self.model = create_model(config.model, N_POSES, self.POSE_DIM)
        self.model_ema = None
        self.body_model_vis = FLAME(model_path=self.bodymodel_path, num_betas=self.config.data.num_betas, batch_size=50)
        self.body_model_eval = FLAME(model_path=self.bodymodel_path, num_betas=self.config.data.num_betas, batch_size=50)
        for param in self.body_model_vis.parameters():
            param.requires_grad = False
        for param in self.body_model_eval.parameters():
            param.requires_grad = False
        self.normalize_fn = None
        self.denormalize_fn = None

        # Setup SDEs and functions
        self.sde = self.setup_sde(config)
        self.sampling_shape = (config.eval.batch_size, N_POSES * self.POSE_DIM)
        self.sampling_eps = 1e-3
        self.train_step_fn = None
        self.sampling_fn = None
        self.likelihood_fn = likelihood.get_likelihood_fn(self.sde, lambda x: x, rtol=1e-4, atol=1e-4, eps=1e-4)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def setup(self, stage=None):
        if stage == 'fit':
            self.model_ema = ExponentialMovingAverage(self.model.parameters(),
                                                      decay=config.model.ema_rate,
                                                      device=self.device)
            Normalizer = Posenormalizer(
                data_path=self.data_path,
                normalize=self.config.data.normalize,
                min_max=self.config.data.min_max,
                rot_rep=self.config.data.rot_rep,
                device=self.device
            )
            self.normalize_fn = Normalizer.offline_normalize
            self.denormalize_fn = Normalizer.offline_denormalize
            self.train_step_fn = self.setup_step_fn(config)
            self.sampling_fn = sampling.get_sampling_fn(config, self.sde, self.sampling_shape,
                                                        lambda x: x, self.sampling_eps, self.device)

    def setup_step_fn(self, config):
        # Build one-step training and evaluation functions
        kwargs = {}
        if config.training.auxiliary_loss:
            body_model_train = FLAME(model_path=self.bodymodel_path,
                                     num_betas=100,
                                     batch_size=config.training.batch_size,).to(self.device)
            for param in body_model_train.parameters():
                param.requires_grad = False
            aux_params = {'denormalize': self.denormalize_fn, 'body_model': body_model_train,
                          'model_type': "face-betas", 'denoise_steps': config.training.denoise_steps}
            kwargs.update(aux_params)
        if config.training.random_mask:
            mask_params = {'min_mask_rate': config.training.min_mask_rate,
                           'max_mask_rate': config.training.max_mask_rate,
                           'observation_type': config.training.observation_type}
            kwargs.update(mask_params)

        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting
        return losses.get_step_fn(self.sde, train=True, optimize_fn=optimize_fn,
                                  reduce_mean=config.training.reduce_mean, continuous=continuous,
                                  likelihood_weighting=likelihood_weighting,
                                  auxiliary_loss=config.training.auxiliary_loss,  # auxiliary loss
                                  random_mask=config.training.random_mask,  # ambient Diffusion
                                  **kwargs)

    def setup_sde(self, config):
        # Setup SDEs
        if config.training.sde.lower() == 'vpsde':
            return sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                 N=config.model.num_scales)
        elif config.training.sde.lower() == 'subvpsde':
            return sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                    N=config.model.num_scales)
        elif config.training.sde.lower() == 'vesde':
            return sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                                 N=config.model.num_scales)
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    def training_step(self, batch, batch_idx):
        face_params = batch['betas']
        face_params = self.normalize_fn(face_params, from_axis=True)
        # Forward pass and calculate loss
        loss_dict = self.train_step_fn(self.model, batch=face_params, condition=None, mask=None)

        # Log the losses
        for key, value in loss_dict.items():
            self.log(f"{key}", value, prog_bar=True, logger=True)

        return loss_dict['loss']

    def on_train_batch_end(self, *args):
        self.model_ema.update(self.model.parameters())

    def on_validation_epoch_start(self) -> None:
        # Store and copy EMA parameters for validation
        self.model_ema.store(self.model.parameters())
        self.model_ema.copy_to(self.model.parameters())

    def validation_step(self, batch, batch_idx):
        face_params = batch['betas']
        face_params = self.normalize_fn(face_params, from_axis=True)
        # Process the batch and calculate metrics
        eval_metrics, trajs, samples = self.process_validation_batch(face_params)
        self.all_samples.append(samples)

        # Store trajs of the last batch
        self.last_trajs = trajs

        # Log calculated metrics
        for metric_name, metric_value in eval_metrics.items():
            self.log(f'val_{metric_name}', metric_value, sync_dist=True, logger=True)

        return eval_metrics

    def on_validation_epoch_end(self) -> None:
        self.model_ema.restore(self.model.parameters())
        '''     ******* Compute APD and SI *******     '''
        all_results = torch.cat(self.all_samples, dim=0)[:50]
        face_params = self.denormalize_fn(all_results, to_axis=True)
        body_out = self.body_model_eval(betas=face_params)
        joints3d = body_out.Jtr
        APD = average_pairwise_distance(joints3d)
        SI = self_intersections_percentage(body_out.v, body_out.f).mean()
        self.log('APD', APD.item(), sync_dist=True, logger=True)
        self.log('SI', SI.item(), sync_dist=True, logger=True)

        if self.config.training.render:
            # Use the stored trajs and all_results of the last batch
            self.render_and_log_images(self.last_trajs, all_results)

        # Reset the list of samples
        self.all_samples = []

    @torch.no_grad()
    def process_validation_batch(self, poses):
        eval_metrics = {}

        '''     ******* task1 bpd *******     '''
        bpd, z, nfe = self.likelihood_fn(self.model, poses, condition=None)
        eval_metrics['bpd'] = bpd.mean().item()

        '''      ******* task3 generation *******     '''
        trajs, samples = self.sampling_fn(
            self.model,
            observation=None, gather_traj=True,
        )  # [t, b, j*6], [b, j*6]

        return eval_metrics, trajs, samples

    def render_and_log_images(self, trajs, all_results):
        bg_img = np.ones([256, 256, 3]) * 255  # background canvas
        focal = [5000, 5000]
        princpt = [128, 115]

        # Sample some frames for visualization
        slice_step = self.sde.N // 10
        trajs = self.denormalize_fn(trajs[::slice_step, :5, ], to_axis=True).reshape(50, -1)  # [10time, 5sample, j*6]
        all_results = self.denormalize_fn(all_results[:50], to_axis=True)  # [50, j*6]

        # Process and log trajs
        self.process_and_log_meshes(trajs, bg_img, focal, princpt, 'trajs')

        # Process and log samples
        self.process_and_log_meshes(all_results, bg_img, focal, princpt, 'samples')

    def process_and_log_meshes(self, poses, bg_img, focal, princpt, tag_prefix):
        body_out = self.body_model_vis(betas=poses)
        meshes = body_out.v.detach().cpu().numpy()
        faces = body_out.f.cpu().numpy()

        rendered_images = []
        for mesh in meshes:
            rendered_img = render_mesh(bg_img, mesh, faces, {'focal': focal, 'princpt': princpt},
                                       view='front')
            rendered_img_tensor = self.convert_to_tensor(rendered_img)
            rendered_images.append(rendered_img_tensor)

        # Create an image grid and log it
        image_grid = torchvision.utils.make_grid(rendered_images, nrow=10)  # 10 columns
        self.logger.experiment.add_image(f'{tag_prefix}_grid', image_grid, self.current_epoch)

    def convert_to_tensor(self, img):
        # Convert the image to a PyTorch tensor and normalize it to [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_tensor

    def configure_optimizers(self):
        # Set up the optimizer
        optimizer = self.get_optimizer(self.config, self.model.parameters())

        # Set up the learning rate scheduler
        if self.config.optim.warmup > 0:
            lr_scheduler = {
                # 'scheduler': optim.lr_scheduler.LambdaLR(
                #     optimizer,
                #     lr_lambda=lambda step: np.minimum(step / self.config.optim.warmup, 1.0)
                # ),
                'scheduler': CosineWarmupScheduler(optimizer,
                                                   self.config.optim.warmup, self.config.training.n_iters),
                'interval': 'step',
            }
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def get_optimizer(self, config, params):
        if config.optim.optimizer == 'Adam':
            return optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                              eps=config.optim.eps, weight_decay=config.optim.weight_decay)
        elif config.optim.optimizer == 'AdamW':
            return optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.98),
                               eps=config.optim.eps, weight_decay=config.optim.weight_decay)
        elif config.optim.optimizer == 'RAdam':
            return optim.RAdam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                               eps=config.optim.eps, weight_decay=config.optim.weight_decay)
        else:
            raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_ema'] = self.model_ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self.model_ema.load_state_dict(checkpoint['model_ema'])


def main(args, config, try_resume):
    pl.seed_everything(config.seed)
    config.name = args.name
    data_path = os.path.join(args.data_root, 'betas_normalizer')

    # Initialize the PyTorch Lightning data module and model
    data_module = FlameDataModule(config, args)
    data_module.setup(stage='fit')
    N_POSES = config.data.num_betas   # 100 / 300 shape coefficients
    if N_POSES == 300:
        data_path = os.path.join(args.data_root, 'betas300_normalizer')
    model = DPoserTrainer(config, args.bodymodel_path, data_path, N_POSES,
                          train_loader=data_module.train_dataloader(),
                          val_loader=data_module.val_dataloader(),)

    # Define logger and callbacks
    logger = TensorBoardLogger(f"logs/dposer/{config.dataset}", name=args.name)
    ckpt_dir = f"checkpoints/dposer/{config.dataset}/{args.name}"
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                          filename='{epoch:02d}-{step}-{loss:.2f}',
                                          every_n_train_steps=config.training.save_freq,
                                          save_top_k=3, save_last=True, monitor='loss', mode='min')
    model_logger = ModelSizeCallback()
    time_monitor = TimerCallback()
    lr_monitor = LearningRateMonitor()

    # Resume training
    resume_from_checkpoint = None
    if args.resume_ckpt is not None:
        resume_from_checkpoint = os.path.join(ckpt_dir, args.resume_ckpt)
        print('Resuming the training from {}'.format(resume_from_checkpoint))
    elif try_resume:
        available_ckpts = os.path.join(ckpt_dir, 'last.ckpt')
        if os.path.exists(available_ckpts):
            resume_from_checkpoint = os.path.realpath(available_ckpts)
            print('Resuming the training from {}'.format(resume_from_checkpoint))

    # Initialize the trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=config.devices,
        strategy='ddp',
        max_steps=config.training.n_iters,
        num_sanity_val_steps=1,
        val_check_interval=config.training.eval_freq,
        check_val_every_n_epoch=None,
        log_every_n_steps=config.training.log_freq,
        gradient_clip_val=config.optim.grad_clip,
        logger=logger,
        callbacks=[model_logger, time_monitor, lr_monitor, checkpoint_callback],
        benchmark=True,
        limit_val_batches=1,
    )

    # Train the model
    trainer.fit(model, ckpt_path=resume_from_checkpoint)


if __name__ == '__main__':
    args = parse_args(sys.argv)
    config = import_configs(args.config_path)
    # there seems to be a bug in PyTorch Lightning while loading EMA parameters from resume.
    resume_training_if_possible = False
    main(args, config, resume_training_if_possible)
