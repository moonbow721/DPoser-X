import os
import random

import torch
import torch.nn.functional as F
import numpy as np

from lib.body_model.utils import BodyPartIndices, HandPartIndices, OpHandPartIndices, FacePartIndices
from lib.body_model import constants
from lib.utils.transforms import rot6d_to_axis_angle


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def add_noise(gts, std=0.5, noise_type='gaussian'):
    if std == 0.0:
        return gts

    if noise_type == 'gaussian':
        noise = std * torch.randn(*gts.shape, device=gts.device)
        gts = gts + noise
    elif noise_type == 'uniform':
        # a range of [-0.5std, 0.5std]
        noise = std * (torch.rand(*gts.shape, device=gts.device) - 0.5)
        gts = gts + noise
    else:
        raise NotImplementedError
    return gts


def create_mask(body_poses, part='legs', model='body', observation_type='noise'):
    # body_poses: [batchsize, 3*N_POSES] (axis-angle) or [batchsize, 6*N_POSES] (rot6d)
    if model == 'body':
        N_POSES = 21
        PartIndices = BodyPartIndices
    elif model == 'hand':
        N_POSES = 15
        PartIndices = HandPartIndices
    else:
        raise ValueError(f'Unknown model: {model}')
    assert len(body_poses.shape) == 2 and body_poses.shape[1] % N_POSES == 0
    rot_N = body_poses.shape[1] // N_POSES
    assert rot_N in [3, 6]

    mask_joints = PartIndices.get_indices(part)
    mask = body_poses.new_ones(body_poses.shape, dtype=torch.bool)
    mask_indices = torch.tensor(mask_joints, dtype=torch.long).view(-1, 1) * rot_N + torch.arange(rot_N).view(1, -1)
    mask_indices = mask_indices.flatten()
    mask[:, mask_indices] = 0

    # masked data as Gaussian noise
    observation = body_poses.clone()
    if observation_type == 'noise':
        observation[:, mask_indices] = torch.randn_like(observation[:, mask_indices])
    elif observation_type == 'zeros':
        observation[:, mask_indices] = torch.zeros_like(observation[:, mask_indices])
    # load the mean pose as observation
    else:
        batch_size = body_poses.shape[0]
        smpl_mean_params = np.load(constants.SMPL_MEAN_PATH)
        rot6d_body_poses = torch.tensor(smpl_mean_params['pose'][6:,], dtype=torch.float32, device=body_poses.device)  # [138]
        axis_body_pose = rot6d_to_axis_angle(rot6d_body_poses.reshape(-1, 6)).reshape(-1)   # [69]
        if rot_N == 3:
            observation[:, mask_indices] = axis_body_pose[None, mask_indices].repeat(batch_size, 1)
        elif rot_N == 6:
            observation[:, mask_indices] = rot6d_body_poses[None, mask_indices].repeat(batch_size, 1)
        else:
            raise NotImplementedError

    return mask, observation


def create_wholebody_mask(body_poses, part='lhand', observation_type='noise', mask_root=True):
    # body_poses: [batchsize, 3*N_POSES+100] (axis-angle) or [batchsize, 6*N_POSES+100] (rot6d)
    N_POSES = 21+15*2+1
    assert len(body_poses.shape) == 2 and (body_poses.shape[1]-100) % N_POSES == 0, body_poses.shape
    rot_N = (body_poses.shape[1]-100) // N_POSES
    assert rot_N in [3, 6], body_poses.shape

    mask = body_poses.new_ones(body_poses.shape, dtype=torch.bool)
    mask_type = part
    if part == 'lhand':
        mask_indices = list(range(21*rot_N, (21+15)*rot_N))
        if mask_root:
            mask_indices += list(range(19*rot_N, 20*rot_N))
    elif part == 'rhand':
        mask_indices = list(range((21+15)*rot_N, (21+15*2)*rot_N))
        if mask_root:
            mask_indices += list(range(20*rot_N, 21*rot_N))
    elif part == 'face':
        mask_indices = list(range((21+15*2)*rot_N, (21+15*2+1)*rot_N + 100))
    elif part == 'one_hand':
        # mask one hand randomly
        if random.random() < 0.5:
            mask_type = 'lhand'
            mask_indices = list(range(21*rot_N, (21+15)*rot_N))
            if mask_root:
                mask_indices += list(range(19*rot_N, 20*rot_N))
        else:
            mask_type = 'rhand'
            mask_indices = list(range((21+15)*rot_N, (21+15*2)*rot_N))
            if mask_root:
                mask_indices += list(range(20*rot_N, 21*rot_N))
    else:
        raise ValueError(f'Unknown part: {part}')
    mask[:, mask_indices] = 0

    # masked data as Gaussian noise
    observation = body_poses.clone()
    if observation_type == 'noise':
        observation[:, mask_indices] = torch.randn_like(observation[:, mask_indices])
    else:
        observation[:, mask_indices] = torch.zeros_like(observation[:, mask_indices])

    return mask, observation, mask_type


def create_random_mask(body_poses, min_mask_rate=0.2, max_mask_rate=0.4, model='body', observation_type='noise'):
    # body_poses: [batchsize, 3*N_POSES] (axis-angle) or [batchsize, 6*N_POSES] (rot6d)
    if model == 'body':
        N_POSES = 21
    elif model == 'hand':
        N_POSES = 15
    else:
        raise ValueError(f'Unknown model: {model}')
    assert len(body_poses.shape) == 2 and body_poses.shape[1] % N_POSES == 0
    rot_N = body_poses.shape[1] // N_POSES
    assert rot_N in [3, 6]

    mask_rate = random.uniform(min_mask_rate, max_mask_rate)
    num_joints_to_mask = int(round(mask_rate * N_POSES))
    if num_joints_to_mask == 0:
        return body_poses.new_ones(body_poses.shape), body_poses
    mask_joints = random.sample(range(N_POSES), num_joints_to_mask)
    mask = body_poses.new_ones(body_poses.shape, dtype=torch.bool)
    mask_indices = torch.tensor(mask_joints).view(-1, 1) * rot_N + torch.arange(rot_N).view(1, -1)
    mask_indices = mask_indices.flatten()
    mask[:, mask_indices] = 0

    # masked data as Gaussian noise
    observation = body_poses.clone()
    if observation_type == 'noise':
        observation[:, mask_indices] = torch.randn_like(observation[:, mask_indices])
    else:
        observation[:, mask_indices] = torch.zeros_like(observation[:, mask_indices])

    return mask, observation


def create_random_part_mask(batchsize, device, mask_prob=0.3, rot_N=3, apply_loss=True,):
    # mask_prob: the probability of masking a part,
    # part_mask: [batchsize, 4], loss_mask: [batchsize, 52*rot_N+100]
    loss_indices = [list(range(21*rot_N)), list(range(21*rot_N, 36*rot_N)),
                    list(range(36*rot_N, 51*rot_N)), list(range(51*rot_N, 52*rot_N+100))]
    part_mask = torch.rand(batchsize, 4, device=device) < mask_prob
    loss_mask = torch.ones(batchsize, 52*rot_N+100, dtype=torch.bool).to(device)
    if not apply_loss:
        for idx in range(4):
            loss_mask[:, loss_indices[idx]] = part_mask[:, idx].unsqueeze(1)

    return part_mask, loss_mask


def apply_random_part_mask(original_mask, mask_prob=0.3, rot_N=3, apply_loss=True):
    # mask_prob: the probability of masking a part,
    # original_mask: [batchsize, 4], dtype=torch.bool
    # part_mask: [batchsize, 4], loss_mask: [batchsize, 52*rot_N+100]
    batchsize = original_mask.shape[0]
    loss_indices = [list(range(21 * rot_N)), list(range(21 * rot_N, 36 * rot_N)),
                    list(range(36 * rot_N, 51 * rot_N)), list(range(51 * rot_N, 52 * rot_N + 100))]
    loss_mask = torch.ones(batchsize, 52 * rot_N + 100, dtype=torch.bool).to(original_mask.device)
    for idx in range(4):
        loss_mask[:, loss_indices[idx]] = original_mask[:, idx].unsqueeze(1)
    if mask_prob == 0.0:
        return original_mask, loss_mask

    # mask the fully_visible samples with mask_prob
    fully_visible = torch.all(original_mask, dim=1)
    combined_mask = original_mask.clone()
    combined_mask[fully_visible] = torch.rand(fully_visible.sum(), 4, device=original_mask.device) < mask_prob

    if not apply_loss:  # apply the mask to the loss_mask for random part masking
        for idx in range(4):
            loss_mask[:, loss_indices[idx]] = combined_mask[:, idx].unsqueeze(1)

    return combined_mask, loss_mask


def create_joint_mask(body_joints, part='legs', mask_type=None, model='body'):
    # body_joints: [batchsize, 22, 3]
    if model == 'body':
        N_JOINTS = 22
        PartIndices = BodyPartIndices
    elif model == 'hand':
        N_JOINTS = 16
        PartIndices = HandPartIndices
    elif model == 'face':
        N_JOINTS = 73
        PartIndices = FacePartIndices
    else:
        raise ValueError(f'Unknown model: {model}')
    assert len(body_joints.shape) == 3 and body_joints.shape[1] == N_JOINTS

    mask_indices = PartIndices.create_mask(mask_type) if mask_type is not None else PartIndices.get_joint_indices(part)
    visible_indices = [i for i in range(N_JOINTS) if i not in mask_indices]
    mask = body_joints.new_ones(body_joints.shape, dtype=torch.bool)
    mask_indices = torch.tensor(mask_indices)
    mask[:, mask_indices, :] = False

    # masked data as Gaussian noise
    observation = body_joints.clone()
    observation[:, mask_indices] = torch.randn_like(observation[:, mask_indices]) * 0.1

    return mask, observation, mask_indices, visible_indices


def create_OpJtr_mask(body_joints, part='index_finger', mask_type=None, model='hand'):
    # hand_OpJtr: [batchsize, 21, 3]
    if model == 'hand':
        N_JOINTS = 21
        PartIndices = OpHandPartIndices
    else:
        raise ValueError(f'Unknown model: {model}')
    assert len(body_joints.shape) == 3 and body_joints.shape[1] == N_JOINTS, body_joints.shape

    mask_indices = PartIndices.create_mask(mask_type) if mask_type is not None else PartIndices.get_indices(part)
    visible_indices = [i for i in range(N_JOINTS) if i not in mask_indices]
    mask = body_joints.new_ones(body_joints.shape, dtype=torch.bool)
    mask_indices = torch.tensor(mask_indices)
    mask[:, mask_indices, :] = False

    # masked data as Gaussian noise
    observation = body_joints.clone()
    observation[:, mask_indices] = torch.randn_like(observation[:, mask_indices]) * 0.05

    return mask, observation, mask_indices, visible_indices


def create_stable_mask(mask, eps=1e-6):
    stable_mask = torch.where(mask,
                              torch.tensor(1.0, dtype=torch.float32, device=mask.device),
                              torch.tensor(eps, dtype=torch.float32, device=mask.device))

    return stable_mask


def lerp(A, B, steps):
    A, B = A.unsqueeze(0), B.unsqueeze(0)
    alpha = torch.linspace(0, 1, steps, device=A.device)
    while alpha.dim() < A.dim():
        alpha = alpha.unsqueeze(-1)

    interpolated = (1 - alpha) * A + alpha * B
    return interpolated


def slerp(v1, v2, steps, DOT_THR=0.9995, zdim=-1):
    """
    SLERP for pytorch tensors interpolating `v1` to `v2` over `num_frames`.

    :param v1: Start vector.
    :param v2: End vector.
    :param num_frames: Number of frames in the interpolation.
    :param DOT_THR: Threshold for parallel vectors.
    :param zdim: Dimension over which to compute norms and find angles.
    :return: Interpolated frames.
    """
    # Normalize the input vectors
    v1_norm = v1 / torch.norm(v1, dim=zdim, keepdim=True)
    v2_norm = v2 / torch.norm(v2, dim=zdim, keepdim=True)

    # Dot product
    dot = (v1_norm * v2_norm).sum(zdim)

    # Mask for vectors that are too close to parallel
    parallel_mask = torch.abs(dot) > DOT_THR

    # SLERP interpolation
    theta = torch.acos(dot).unsqueeze(0)
    alpha = torch.linspace(0, 1, steps, device=v1.device)
    while alpha.dim() < theta.dim():
        alpha = alpha.unsqueeze(-1)
    theta_t = theta * alpha
    sin_theta = torch.sin(theta)
    sin_theta_t = torch.sin(theta_t)

    s1 = torch.sin(theta - theta_t) / sin_theta
    s2 = sin_theta_t / sin_theta
    slerp_res = (s1.unsqueeze(zdim) * v1) + (s2.unsqueeze(zdim) * v2)

    # LERP interpolation
    lerp_res = lerp(v1, v2, steps)

    # Combine results based on the parallel mask
    combined_res = torch.where(parallel_mask.unsqueeze(zdim), lerp_res, slerp_res)

    return combined_res


def moving_average(data, window_size):
    kernel = torch.ones(window_size) / window_size
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(data.device)

    data = data.transpose(0, 1).unsqueeze(1)

    smoothed_data = F.conv1d(data, kernel, padding=window_size//2)

    smoothed_data = smoothed_data.squeeze(1).transpose(0, 1)
    return smoothed_data


def gaussian_smoothing(data, window_size, sigma):
    kernel = torch.arange(window_size).float() - window_size // 2
    kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
    kernel /= kernel.sum()

    kernel = kernel.unsqueeze(0).unsqueeze(0).to(data.device)
    data = data.transpose(0, 1).unsqueeze(1)

    smoothed_data = F.conv1d(data, kernel, padding=window_size//2)

    smoothed_data = smoothed_data.squeeze(1).transpose(0, 1)
    return smoothed_data


def rbf_kernel(X, Y, gamma=-1, ad=1):
    # X and Y should be tensors with shape (batch_size, data_dim)
    # gamma is a hyperparameter controlling the width of the RBF kernel

    # Compute the pairwise squared Euclidean distances between the samples
    with torch.cuda.amp.autocast():
        dists = torch.cdist(X, Y, p=2) ** 2

    if gamma < 0:  # use median trick
        gamma = torch.median(dists)
        gamma = torch.sqrt(0.5 * gamma / np.log(X.size(0) + 1))
        gamma = 1 / (2 * gamma ** 2)
    else:
        gamma = gamma * ad

    # Compute the RBF kernel using the squared distances and gamma
    K = torch.exp(-gamma * dists)

    # Compute the gradient of the RBF kernel with respect to X
    dK = -2 * gamma * K.unsqueeze(-1) * (X.unsqueeze(1) - Y.unsqueeze(0))
    dK_dX = torch.sum(dK, dim=1)

    return K, dK_dX
