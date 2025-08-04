import functools

import torch
import torch.nn as nn

from lib.algorithms.advanced.module import GaussianFourierProjection, get_sigmas, get_timestep_embedding, get_act


def create_model(model_config, N_POSES, POSE_DIM):
    if 'FC' in model_config.type:
        model = TimeFC(
            model_config,
            n_poses=N_POSES,
            pose_dim=POSE_DIM,
            hidden_dim=model_config.HIDDEN_DIM,
            embed_dim=model_config.EMBED_DIM,
            n_blocks=model_config.N_BLOCKS,
        )
    elif model_config.type == 'TimeMLPs':
        model = TimeMLPs(
            model_config,
            n_poses=N_POSES,
            pose_dim=POSE_DIM,
            hidden_dim=model_config.HIDDEN_DIM,
            n_blocks=model_config.N_BLOCKS,
        )
    else:
        raise NotImplementedError('unsupported model')

    return model


class TimeMLPs(torch.nn.Module):
    def __init__(self, config, n_poses=21, pose_dim=6, hidden_dim=64, n_blocks=2):
        super().__init__()
        dim = n_poses * pose_dim
        self.act = get_act(config)

        layers = [torch.nn.Linear(dim + 1, hidden_dim),
                  self.act]

        for _ in range(n_blocks):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                self.act,
                torch.nn.Dropout(p=config.dropout)
            ])

        layers.append(torch.nn.Linear(hidden_dim, dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, batch, t, condition=None, mask=None):
        return self.net(torch.cat([batch, t[:, None]], dim=1))


class TimeFC(nn.Module):
    """
    Independent condition feature projection layers for each block
    """

    def __init__(self, model_config, n_poses=21, pose_dim=6, hidden_dim=64,
                 embed_dim=32, n_blocks=2):
        super(TimeFC, self).__init__()
        self.model_config = model_config
        self.n_poses = n_poses
        self.joint_dim = pose_dim
        self.n_blocks = n_blocks

        self.act = get_act(model_config)

        self.pre_dense = nn.Linear(n_poses * pose_dim, hidden_dim)
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

        self.post_dense = nn.Linear(hidden_dim, n_poses * pose_dim)

    def forward(self, batch, t, condition=None, mask=None):
        """
        batch: [B, j*3] or [B, j*6]
        t: [B]
        condition: not be enabled
        mask: [B, j*3] or [B, j*6] same dim as batch
        Return: [B, j*3] or [B, j*6] same dim as batch
        """
        bs = batch.shape[0]

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

        h = self.pre_dense(batch)
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