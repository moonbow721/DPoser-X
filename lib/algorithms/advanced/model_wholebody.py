import functools

import torch
import torch.nn as nn

from lib.algorithms.advanced.model import create_model
from lib.utils.generic import load_model, import_configs
from lib.algorithms.advanced.module import GaussianFourierProjection, get_sigmas, get_timestep_embedding, get_act


def create_wholebody_model(model_config, POSES_LIST, POSE_DIM):
    if model_config.type == 'Combiner':
        model = Combine_wholebody_model(
            import_configs(model_config.body_config).model,
            import_configs(model_config.hand_config).model,
            import_configs(model_config.face_config).model,
            poses_list=POSES_LIST,
            pose_dim=POSE_DIM,
            body_ckpt=model_config.body_ckpt,
            hand_ckpt=model_config.hand_ckpt,
            face_ckpt=model_config.face_ckpt,
        )
    elif model_config.type == 'Finetune':
        model = Finetune_wholebody_model(
            import_configs(model_config.body_config).model,
            import_configs(model_config.hand_config).model,
            import_configs(model_config.face_config).model,
            model_config,
            poses_list=POSES_LIST,
            pose_dim=POSE_DIM,
            body_ckpt=model_config.body_ckpt,
            hand_ckpt=model_config.hand_ckpt,
            face_ckpt=model_config.face_ckpt,
        )
    else:
        raise NotImplementedError('unsupported model')

    return model


class Combine_wholebody_model(nn.Module):
    """
    Independent condition feature projection layers for each block
    """

    def __init__(self, body_config, hand_config, face_config,
                 poses_list, pose_dim, body_ckpt, hand_ckpt, face_ckpt, ):
        super(Combine_wholebody_model, self).__init__()
        self.body_posenum = poses_list[0]
        self.hand_posenum = poses_list[1]
        self.face_posenum = poses_list[2]
        self.pose_dim = pose_dim
        self.body_model = create_model(body_config, poses_list[0], pose_dim)
        self.hand_model = create_model(hand_config, poses_list[1], pose_dim)    # right hand
        self.face_model = create_model(face_config, poses_list[2], 1)
        load_model(self.body_model, body_config, body_ckpt, 'cpu', is_ema=True)
        load_model(self.hand_model, hand_config, hand_ckpt, 'cpu', is_ema=True)
        load_model(self.face_model, face_config, face_ckpt, 'cpu', is_ema=True)

    def forward(self, batch, t, condition_list=None, mask_list=None):
        """
        batch: [B, j*3] or [B, j*6], Order: [body, left_hand, right_hand, face]
        t: [B]
        condition: not be enabled
        mask: [B, j*3] or [B, j*6] same dim as batch
        Return: [B, j*3] or [B, j*6] same dim as batch
        """
        if condition_list is None:
            condition_list = [None, None, None, None]
        if mask_list is None:
            mask_list = [None, None, None, None]
        body_pose, left_hand_pose, right_hand_pose, face_pose = (
            torch.split(batch, [self.body_posenum * self.pose_dim,
                                self.hand_posenum * self.pose_dim,
                                self.hand_posenum * self.pose_dim,
                                self.face_posenum * 1], dim=1))
        # The left_hand_pose should already be flipped as preprocessing
        output_body = self.body_model(body_pose, t, condition_list[0], mask_list[0])
        output_left_hand = self.hand_model(left_hand_pose, t, condition_list[1], mask_list[1])
        output_right_hand = self.hand_model(right_hand_pose, t, condition_list[2], mask_list[2])
        output_face = self.face_model(face_pose, t, condition_list[3], mask_list[3])

        return torch.cat([output_body, output_left_hand, output_right_hand, output_face], dim=1)


class Finetune_wholebody_model(nn.Module):
    """
    Independent condition feature projection layers for each block
    """

    def __init__(self, body_config, hand_config, face_config, wholebody_config,
                 poses_list, pose_dim, body_ckpt, hand_ckpt, face_ckpt, ):
        super(Finetune_wholebody_model, self).__init__()
        self.body_posenum = poses_list[0]
        self.hand_posenum = poses_list[1]
        self.face_posenum = poses_list[2]
        self.pose_dim = pose_dim
        full_dim = poses_list[0] * pose_dim + poses_list[1] * 2 * pose_dim + poses_list[2]
        self.base_model = Combine_wholebody_model(
            body_config, hand_config, face_config, poses_list, pose_dim, body_ckpt, hand_ckpt, face_ckpt)
        self.body_hidden_dim = body_config.HIDDEN_DIM
        self.hand_hidden_dim = hand_config.HIDDEN_DIM
        self.face_hidden_dim = face_config.HIDDEN_DIM
        # Fix the base model weights
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.mask_fc = MaskFC(wholebody_config, full_dim,
                              [self.body_hidden_dim, self.hand_hidden_dim, self.face_hidden_dim],
                              wholebody_config.HIDDEN_DIM, wholebody_config.EMBED_DIM, wholebody_config.N_BLOCKS)
        self.output_linear = nn.Linear(full_dim, full_dim)
        self.zero_linear = self.zero_module(self.output_linear)

    def zero_module(self, module):
        for p in module.parameters():
            nn.init.zeros_(p)
        return module

    def forward(self, batch, t, condition_list=None, mask=None):
        """
        batch: [B, j*3] or [B, j*6], Order: [body, left_hand, right_hand, face]
        t: [B]
        condition_list: not be enabled
        mask: [B, 4], 0 for body, 1 for left_hand, 2 for right_hand, 3 for face
        Return: [B, j*3] or [B, j*6] same dim as batch
        """
        if condition_list is None:
            condition_list = [None, None, None, None]
        if mask is None:
            mask = torch.ones(batch.shape[0], 4).to(batch.device)
        body_pose, left_hand_pose, right_hand_pose, face_pose = (
            torch.split(batch, [self.body_posenum * self.pose_dim,
                                self.hand_posenum * self.pose_dim,
                                self.hand_posenum * self.pose_dim,
                                self.face_posenum * 1], dim=1))
        # If some parts are masked, pass through the base model with the mean value (zeros after normalization)
        body_pose = body_pose * mask[:, 0][:, None]
        left_hand_pose = left_hand_pose * mask[:, 1][:, None]
        right_hand_pose = right_hand_pose * mask[:, 2][:, None]
        face_pose = face_pose * mask[:, 3][:, None]
        batch = torch.cat([body_pose, left_hand_pose, right_hand_pose, face_pose], dim=1)

        base_result = self.base_model(batch, t, condition_list, [None, None, None, None])
        hidden_body, hidden_lhand, hidden_rhand, hidden_face = \
        (self.base_model.body_model.pre_dense(body_pose), self.base_model.hand_model.pre_dense(left_hand_pose),
         self.base_model.hand_model.pre_dense(right_hand_pose), self.base_model.face_model.pre_dense(face_pose))

        output = self.mask_fc(hidden_body, hidden_lhand, hidden_rhand, hidden_face, t, condition_list, mask)
        return base_result + self.zero_linear(output)


class MaskFC(nn.Module):
    def __init__(self, model_config, full_dim, input_dim_list=None,
                 hidden_dim=64, embed_dim=32, n_blocks=2):
        super(MaskFC, self).__init__()
        if input_dim_list is None:
            input_dim_list = [1024, 1024, 1024]
        self.input_dim_list = input_dim_list
        self.input_feat_dim = input_dim_list[0] + input_dim_list[1] * 2 + input_dim_list[2]
        self.full_dim = full_dim
        self.model_config = model_config

        self.n_blocks = n_blocks

        self.act = get_act(model_config)

        self.pre_dense = nn.Linear(self.input_feat_dim, hidden_dim)
        self.pre_dense_t = nn.Linear(embed_dim, hidden_dim)

        self.pre_gnorm = nn.GroupNorm(32, num_channels=hidden_dim)
        self.dropout = nn.Dropout(p=model_config.dropout)

        # time embedding
        self.time_embedding_type = model_config.embedding_type.lower()
        if self.time_embedding_type == 'fourier':
            self.gauss_proj = GaussianFourierProjection(embed_dim=embed_dim, scale=model_config.fourier_scale)
        elif self.time_embedding_type == 'positional':
            self.posit_proj = functools.partial(get_timestep_embedding, embedding_dim=embed_dim)
        else:
            assert 0

        self.shared_time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
        )
        self.register_buffer('sigmas', torch.tensor(get_sigmas(model_config), dtype=torch.float))

        for idx in range(n_blocks):
            setattr(self, f'b{idx + 1}_dense1', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_dense1_t', nn.Linear(embed_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_gnorm1', nn.GroupNorm(32, num_channels=hidden_dim))

            setattr(self, f'b{idx + 1}_dense2', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_dense2_t', nn.Linear(embed_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_gnorm2', nn.GroupNorm(32, num_channels=hidden_dim))

        self.post_dense = nn.Linear(hidden_dim, self.full_dim)


    def forward(self, hidden_body, hidden_lhand, hidden_rhand, hidden_face, t, condition=None, mask=None, ):
        """
        mask: [B, 4], 0 for body, 1 for left_hand, 2 for right_hand, 3 for face
        """
        bs = hidden_body.shape[0]
        # # [b, 4] -> [b, full_dim]
        # mask_tensor = torch.cat([mask[:, 0][:, None].repeat(1, self.input_dim_list[0]),
        #                          mask[:, 1][:, None].repeat(1, self.input_dim_list[1]),
        #                          mask[:, 2][:, None].repeat(1, self.input_dim_list[1]),
        #                          mask[:, 3][:, None].repeat(1, self.input_dim_list[2])], dim=1)
        # # Mask batching, replace masked parts with zeros
        # masked_batch = torch.cat([hidden_body, hidden_lhand, hidden_rhand, hidden_face], dim=1) * mask_tensor

        batch = torch.cat([hidden_body, hidden_lhand, hidden_rhand, hidden_face], dim=1)

        # time embedding
        if self.time_embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = self.gauss_proj(torch.log(used_sigmas))
        elif self.time_embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = t
            used_sigmas = self.sigmas[t.long()]
            temb = self.posit_proj(timesteps)
        else:
            raise ValueError(f'time embedding type {self.time_embedding_type} unknown.')

        temb = self.shared_time_embed(temb)

        h = self.pre_dense(batch)  # [B, j, hidden_dim]
        h += self.pre_dense_t(temb)
        h = self.pre_gnorm(h)
        h = self.act(h)
        h = self.dropout(h)

        for idx in range(self.n_blocks):
            h1 = getattr(self, f'b{idx + 1}_dense1')(h)
            h1 += getattr(self, f'b{idx + 1}_dense1_t')(temb)
            h1 = getattr(self, f'b{idx + 1}_gnorm1')(h1)
            h1 = self.act(h1)
            # dropout, maybe
            h1 = self.dropout(h1)

            h2 = getattr(self, f'b{idx + 1}_dense2')(h1)
            h2 += getattr(self, f'b{idx + 1}_dense2_t')(temb)
            h2 = getattr(self, f'b{idx + 1}_gnorm2')(h2)
            h2 = self.act(h2)
            # dropout, maybe
            h2 = self.dropout(h2)

            h = h + h2

        res = self.post_dense(h)  # [B, j*3]

        ''' normalize the output '''
        if self.model_config.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((bs, 1))
            res = res / used_sigmas

        return res