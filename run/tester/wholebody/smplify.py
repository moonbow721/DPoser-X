import os
import pickle
import torch
from torch import nn

from lib.algorithms.advanced import sde_lib
from lib.algorithms.advanced import utils as mutils
from lib.algorithms.advanced.model_wholebody import create_wholebody_model
from lib.body_model import constants
from lib.body_model.body_model import fullpose_to_params
from lib.body_model.fitting_losses import camera_fitting_loss, wholebody_fitting_loss
from lib.dataset.utils import CombinedNormalizer
from lib.utils.generic import import_configs, load_pl_weights, load_model
from lib.utils.misc import lerp
from lib.utils.transforms import flip_orientations


class DPoser(nn.Module):
    def __init__(self, batch_size=32, config_path='', args=None):
        super().__init__()
        self.device = args.device
        self.batch_size = batch_size
        config = import_configs(config_path)

        data_path = {"body_pose": os.path.join(args.data_path, 'body_data', 'body_normalizer'),
                     "left_hand_pose": os.path.join(args.data_path, 'hand_data', 'hand_normalizer'),
                     "right_hand_pose": os.path.join(args.data_path, 'hand_data', 'hand_normalizer'),
                     "jaw_pose": os.path.join(args.data_path, 'face_data', 'jaw_normalizer'),
                     "expression": os.path.join(args.data_path, 'face_data', 'expression_normalizer')}
        self.Normalizer = CombinedNormalizer(
            data_path_dict=data_path, model='whole-body',
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
        # L2 loss, set expression as 0.0 for arctic dataset evaluation
        weight_dict = {'body': 1.0, 'hands': 1.0, 'jaw': 1.0, 'expression': 1.0}
        self.part_weights = torch.ones((batch_size, 256), device=self.device)
        self.part_weights[:, :63] = weight_dict['body']
        self.part_weights[:, 63:63 + 90] = weight_dict['hands']
        self.part_weights[:, 153:153 + 3] = weight_dict['jaw']
        self.part_weights[:, 156:] = weight_dict['expression']
        self.loss_fn = nn.MSELoss(reduction='none')

    def load_model(self, config, args):
        POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
        POSES_LIST = [21, 15, 100 + 3]
        if config.data.rot_rep == 'rot6d':
            POSES_LIST[2] = 100 + 6
        DATA_DIM = POSES_LIST[0] * POSE_DIM + 2 * POSES_LIST[1] * POSE_DIM + POSES_LIST[2]  # two hands
        model = create_wholebody_model(config.model, POSES_LIST, POSE_DIM)
        if config.model.type != 'Combiner':
            load_model(model, config.model, args.ckpt_path, args.device, is_ema=True)
        model.to(args.device)
        model.eval()

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
        weight = 0.5*torch.sqrt(1 + SNR ** 2) * self.part_weights
        loss = torch.sum(weight * self.loss_fn(x_0, denoise_data)) / self.batch_size

        return loss

    def forward(self, wholebody_params, betas, t):
        wholebody_params = self.Normalizer.offline_normalize(wholebody_params, from_axis=True)
        vec_t = torch.ones(self.batch_size, device=self.device) * t
        prior_loss = self.DPoser_loss(wholebody_params, vec_t)
        return prior_loss


class SMPLify:
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 body_model,
                 step_size=1e-2,
                 batch_size=32,
                 num_iters=100,
                 focal_length=5000,
                 side_view_thsh=25.0,
                 args=None):
        self.smpl = body_model
        # Store options
        self.device = args.device
        self.focal_length = focal_length
        self.side_view_thsh = side_view_thsh
        self.step_size = step_size

        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip']
        self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]
        self.num_iters = num_iters
        self.prior_name = args.prior

        if args.prior == 'DPoser':
            self.pose_prior = {'wholebody': DPoser(batch_size, args.config_path, args)}
            self.time_strategy = args.time_strategy
            self.t_max = 0.12
            self.t_min = 0.08
            self.fixed_t = 0.10
        else:
            self.pose_prior = {'body': None}

        body_weights = hand_weights = face_weights = [50, 20, 10, 5, 2]
        part_weights = {'wholebody': body_weights, 'body': body_weights,
                        'lhand': hand_weights, 'rhand': hand_weights, 'face': face_weights, }
        part_weights = [dict(zip(part_weights.keys(), vals)) for vals in zip(*part_weights.values())]
        self.loss_weights = {'pose_prior_weight': part_weights,
                             'shape_prior_weight': [50, 20, 10, 5, 2],
                             'expr_prior_weight': [50, 20, 10, 5, 2],
                             'angle_prior_weight': [50, 20, 10, 5, 2],
                             'coll_loss_weight': [0, 0, 0, 0.01, 1.0],
                             }
        self.joint_weights = torch.ones((batch_size, 135), device=self.device)
        self.joint_part_weights = {'hands': [0.0, 0.0, 0.0, 0.2, 0.5],
                                   'face': [0.0, 0.0, 0.0, 0.0, 0.2],
                                   }
        self.stages = len(self.loss_weights['pose_prior_weight'])
        self.interpenetration = args.interpenetration
        self.search_tree, self.pen_distance, self.filter_faces = None, None, None
        self.body_model_faces = self.smpl.bm.faces_tensor.view(-1).to(self.device)
        # not used in our experiments
        if self.interpenetration:
            self.prepare_intersection()

    def sample_continuous_time(self, iteration):
        total_steps = self.stages * self.num_iters
        if self.prior_name == 'DPoser':
            if self.time_strategy == '1':
                t = self.pose_prior['wholebody'].eps + torch.rand(1, device=self.device) * (
                        self.pose_prior['wholebody'].sde.T - self.pose_prior['wholebody'].eps)
            elif self.time_strategy == '2':
                t = torch.tensor(self.fixed_t)
            elif self.time_strategy == '3':
                t = self.t_min + torch.tensor(total_steps - iteration - 1) / total_steps * (self.t_max - self.t_min)
            else:
                raise NotImplementedError
        else:
            t = 0

        return t

    def prepare_intersection(self, max_collisions=128, df_cone_height=0.0001, point2plane=False, penalize_outside=True):
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        self.search_tree = BVH(max_collisions=max_collisions)

        self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
            sigma=df_cone_height, point2plane=point2plane,
            vectorized=True, penalize_outside=penalize_outside)

        # Read the part segmentation
        part_segm_fn = '/data4/ljz24/projects/3DHuman/body_models/smplx/smplx_parts_segm.pkl'
        with open(part_segm_fn, 'rb') as faces_parents_file:
            face_segm_data = pickle.load(faces_parents_file,
                                         encoding='latin1')
        faces_segm = face_segm_data['segm']
        faces_parents = face_segm_data['parents']
        # Create the module used to filter invalid collision pairs
        ign_part_pairs = ["9,16", "9,17", "6,16", "6,17", "1,2", "12,22"]
        self.filter_faces = FilterFaces(
            faces_segm=faces_segm, faces_parents=faces_parents,
            ign_part_pairs=ign_part_pairs).to(device=self.device)

    def __call__(self, init_full_pose, init_expression, init_betas, init_cam_t, camera_center, keypoints_2d):
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
        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2].clone()
        joints_conf = keypoints_2d[:, :, -1].clone()
        # # for arctic dataset
        # joints_conf[:, 19:19+3] = 0.0
        # joints_conf[:, [2,3,5,6]] *= 2.0
        # joints_conf[:, 25:67] = joints_conf[:, 25:67]**2

        # Split SMPL pose to whole-body pose and global orientation
        # [body_pose, left_hand_pose, right_hand_pose, jaw_pose]
        global_orient = init_full_pose[:, :3].detach().clone()
        pose_params = fullpose_to_params(init_full_pose).detach().clone()
        expression = init_expression.detach().clone()
        betas = init_betas.detach().clone()

        # Step 0: Optimize camera translation
        pose_params.requires_grad = False
        betas.requires_grad = False
        expression.requires_grad = False
        global_orient.requires_grad = False
        camera_translation.requires_grad = True
        camera_opt_params = [
            {'params': [camera_translation], 'lr': 0.1}
        ]
        camera_optimizer = torch.optim.Adam(camera_opt_params)
        for i in range(50):
            smpl_output = self.smpl(betas=betas,
                                    wholebody_params=torch.cat([pose_params, expression], dim=-1),
                                    global_orient=global_orient,
                                    trans=camera_translation)

            model_joints = smpl_output.OpJtr
            loss = camera_fitting_loss(model_joints, camera_translation,
                                       init_cam_t, camera_center,
                                       joints_2d, joints_conf, focal_length=self.focal_length, part='body')

            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        pose_params.requires_grad = False
        betas.requires_grad = False
        expression.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]
        camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(100):
            smpl_output = self.smpl(betas=betas,
                                    wholebody_params=torch.cat([pose_params, expression], dim=-1),
                                    global_orient=global_orient,
                                    trans=camera_translation)

            model_joints = smpl_output.OpJtr
            loss = camera_fitting_loss(model_joints, camera_translation,
                                       init_cam_t, camera_center,
                                       joints_2d, joints_conf, focal_length=self.focal_length, part='body')

            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        left_shoulder_idx, right_shoulder_idx = 2, 5
        shoulder_dist = torch.dist(joints_2d[:, left_shoulder_idx],
                                   joints_2d[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < self.side_view_thsh

        # Step 2: Optimize body joints
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        inputs = (
            global_orient, pose_params, expression, betas, camera_translation, camera_center, joints_2d, joints_conf)
        if try_both_orient:
            global_orient, wholebody_params, betas, camera_translation, reprojection_loss = self.optimize_and_compare(
                *inputs)
        else:
            reprojection_loss, (global_orient, wholebody_params, betas, camera_translation) = self.optimize_body(
                *inputs)

        return global_orient, wholebody_params, betas, camera_translation, reprojection_loss

    def optimize_and_compare(self, global_orient, pose_params, expression, betas,
                             camera_translation, camera_center, joints_2d, joints_conf):
        original_loss, original_results = self.optimize_body(global_orient.detach(), pose_params, expression, betas,
                                                             camera_translation, camera_center, joints_2d, joints_conf)
        flipped_loss, flipped_results = self.optimize_body(flip_orientations(global_orient).detach(), pose_params,
                                                           expression, betas,
                                                           camera_translation, camera_center, joints_2d, joints_conf)

        min_loss_indices = original_loss < flipped_loss  # [N,]

        chosen_results = []
        for orig_res, flip_res in zip(original_results, flipped_results):
            selected_res = torch.where(min_loss_indices.unsqueeze(-1), orig_res, flip_res)
            chosen_results.append(selected_res)
        reprojection_loss = torch.where(min_loss_indices, original_loss, flipped_loss)
        chosen_results.append(reprojection_loss)

        return tuple(chosen_results)

    def optimize_body(self, global_orient, pose_params, expression, betas, camera_translation, camera_center, joints_2d,
                      joints_conf):
        """
        Optimize only the body pose and global orientation of the body
        """
        batch_size = global_orient.shape[0]

        global_orient.requires_grad = True
        pose_params.requires_grad = True
        expression.requires_grad = True
        betas.requires_grad = True
        camera_translation.requires_grad = False

        body_opt_params = [global_orient, pose_params, expression, betas, ]

        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        stage_weights = [dict(zip(self.loss_weights.keys(), vals)) for vals in zip(*self.loss_weights.values())]

        for stage, current_weights in enumerate(stage_weights):
            joints_weight = self.joint_weights
            joints_weight[:, 25:67] = self.joint_part_weights['hands'][stage]
            joints_weight[:, 67:] = self.joint_part_weights['face'][stage]

            for i in range(self.num_iters):
                smpl_output = self.smpl(betas=betas,
                                        wholebody_params=torch.cat([pose_params, expression], dim=-1),
                                        global_orient=global_orient,
                                        trans=camera_translation)

                model_joints = smpl_output.OpJtr
                t = self.sample_continuous_time(iteration=stage * self.num_iters + i)

                pen_loss = torch.tensor(0.0).to(device=self.device)
                if (self.interpenetration and current_weights['coll_loss_weight'] > 0):
                    triangles = torch.index_select(smpl_output.v, 1, self.body_model_faces). \
                        view(batch_size, -1, 3, 3)
                    with torch.no_grad():
                        collision_idxs = self.search_tree(triangles)
                    # Remove unwanted collisions
                    if self.filter_faces is not None:
                        collision_idxs = self.filter_faces(collision_idxs)
                    if collision_idxs.ge(0).sum().item() > 0:
                        pen_loss = torch.sum(current_weights['coll_loss_weight'] *
                                             self.pen_distance(triangles, collision_idxs))

                loss = wholebody_fitting_loss(pose_params, expression, betas, model_joints, camera_translation,
                                              camera_center, joints_2d, joints_conf, joints_weight,
                                              pose_prior=self.pose_prior, t=t, focal_length=self.focal_length,
                                              verbose=False, part='body', **current_weights, )
                loss = loss + pen_loss

                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(betas=betas,
                                    wholebody_params=torch.cat([pose_params, expression], dim=-1),
                                    global_orient=global_orient,
                                    trans=camera_translation)

            model_joints = smpl_output.OpJtr
            t = self.sample_continuous_time(iteration=stage * self.num_iters + i)
            reprojection_loss = wholebody_fitting_loss(pose_params, expression, betas, model_joints, camera_translation,
                                                       camera_center, joints_2d, joints_conf, joints_weight,
                                                       pose_prior=self.pose_prior, t=t, focal_length=self.focal_length,
                                                       output='reprojection', verbose=False, part='body',
                                                       **current_weights)

            # for arctic dataset evaluation, since the gt doesn't include face expression
            # pose_params[:, -3:] = torch.zeros_like(pose_params[:, -3:])
            # expression = torch.zeros_like(expression)
        wholebody_params = torch.cat([pose_params, expression], dim=-1).detach()
        return reprojection_loss, (global_orient.detach(), wholebody_params, betas.detach(), camera_translation)
