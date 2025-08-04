from functools import partial

import cv2
import numpy as np
from typing import Dict

import torch
from torch.utils.data import Dataset


class MoCapDataset(Dataset):
    def __init__(self, dataset_file: str):
        """
        Dataset class used for loading a dataset of unpaired MANO parameter annotations
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
        """
        data = np.load(dataset_file)
        self.orient = data['hand_pose'].astype(np.float32)[:, :3]
        self.pose = data['hand_pose'].astype(np.float32)[:, 3:]
        self.betas = data['betas'].astype(np.float32)
        self.length = len(self.pose)

    def __getitem__(self, idx: int) -> Dict:
        orient = self.orient[idx].copy()
        pose = self.pose[idx].copy()
        betas = self.betas[idx].copy()
        item = {'hand_orient': orient, 'hand_pose': pose, 'betas': betas}
        return item

    def __len__(self) -> int:
        return self.length


class ManoDataset(Dataset):
    def __init__(self, dataset_file: str):
        data = np.load(dataset_file)
        self.orient = torch.from_numpy(data['hand_pose'].astype(np.float32))[:, :3]
        self.pose = torch.from_numpy(data['hand_pose'].astype(np.float32))[:, 3:]
        self.betas = torch.from_numpy(data['betas'].astype(np.float32))
        self.length = len(self.pose)

    def __getitem__(self, idx: int) -> Dict:
        orient = self.orient[idx]
        pose = self.pose[idx]
        betas = self.betas[idx]
        item = {'hand_orient': orient, 'hand_pose': pose, 'betas': betas}
        return item

    def __len__(self) -> int:
        return self.length


class InterHandDataset(Dataset):
    def __init__(self, dataset_file: str, single_hand: bool):
        """
        Dataset class for loading a dataset of two-hand parameters.
        Args:
            dataset_file (str): Path to .pt file containing dataset.
            single_hand (bool): If True, combine and shuffle left and right hands.
                                If False, keep them separate.
        """
        data = torch.load(dataset_file)
        self.single_hand = single_hand

        if single_hand:
            # flip the left hand to make it right hand
            data['left_pose'][:, 1::3] *= -1
            data['left_pose'][:, 2::3] *= -1
            left_hands = torch.cat([data['left_pose'], data['left_shape']], dim=1)
            right_hands = torch.cat([data['right_pose'], data['right_shape']], dim=1)
            all_hands = torch.cat([left_hands, right_hands], dim=0)
            self.hands = all_hands

            g = torch.Generator()
            g.manual_seed(42)
            self.hands = all_hands[torch.randperm(all_hands.size(0), generator=g)]
        else:
            self.left_orient = data['left_pose'][..., :3]
            self.left_pose = data['left_pose'][..., 3:]
            self.left_shape = data['left_shape']
            self.right_orient = data['right_pose'][..., :3]
            self.right_pose = data['right_pose'][..., 3:]
            self.right_shape = data['right_shape']

    def __getitem__(self, idx: int):
        if self.single_hand:
            hand_data = self.hands[idx]
            orient, pose, trans = hand_data[..., :3], hand_data[..., 3:-10], hand_data[..., -10:]
            return {'hand_orient': orient, 'hand_pose': pose, 'betas': trans}
        else:
            return {'left_hand_orient': self.left_orient[idx], 'left_hand_pose': self.left_pose[idx],
                    'left_betas': self.left_shape[idx],
                    'right_hand_orient': self.right_orient[idx], 'right_hand_pose': self.right_pose[idx],
                    'right_betas': self.right_shape[idx]}

    def __len__(self):
        return len(self.hands) if self.single_hand else len(self.left_pose)

