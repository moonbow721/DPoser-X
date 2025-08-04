# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import os
import re
from abc import ABC
from functools import reduce, partial
from pathlib import Path

import cv2
import numpy as np
import torch

from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms

from lib.body_model.face_model import FLAME
from lib.utils.misc import to_device


# from loguru import logger
class BaseDataset(Dataset, ABC):
    def __init__(self, name, config, device, isEval):
        self.K = config.K
        self.isEval = isEval
        self.n_train = np.Inf
        self.imagepaths = []
        self.face_dict = {}
        self.name = name
        self.device = device
        self.min_max_K = 0
        self.cluster = False
        self.dataset_root = config.root
        self.total_images = 0
        self.image_folder = 'arcface_input'
        self.flame_folder = 'FLAME_parameters'
        self.initialize()

    def initialize(self):
        logger.info(f'[{self.name}] Initialization')
        image_list = f'{os.path.abspath(os.path.dirname(__file__))}/image_paths/{self.name}.npy'
        logger.info(f'[{self.name}] Load cached file list: ' + image_list)
        self.face_dict = np.load(image_list, allow_pickle=True).item()
        self.imagepaths = list(self.face_dict.keys())
        logger.info(f'[Dataset {self.name}] Total {len(self.imagepaths)} actors loaded!')
        self.set_smallest_k()

    def set_smallest_k(self):
        self.min_max_K = np.Inf
        max_min_k = -np.Inf
        for key in self.face_dict.keys():
            length = len(self.face_dict[key][0])
            if length < self.min_max_K:
                self.min_max_K = length
            if length > max_min_k:
                max_min_k = length

        self.total_images = reduce(lambda k, l: l + k, map(lambda e: len(self.face_dict[e][0]), self.imagepaths))
        logger.info(f'Dataset {self.name} with min K = {self.min_max_K} max K = {max_min_k} length = {len(self.face_dict)} total images = {self.total_images}')
        return self.min_max_K

    def compose_transforms(self, *args):
        self.transforms = transforms.Compose([t for t in args])

    def get_arcface_path(self, image_path):
        return re.sub('png|jpg', 'npy', str(image_path))

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, index):
        actor = self.imagepaths[index]
        images, params_path = self.face_dict[actor]
        images = [Path(self.dataset_root, self.name, self.image_folder, path) for path in images]
        sample_list = np.array(np.random.choice(range(len(images)), size=self.K, replace=False))

        K = self.K
        if self.isEval:
            K = max(0, min(200, self.min_max_K))
            sample_list = np.array(range(len(images))[:K])

        params = np.load(os.path.join(self.dataset_root, self.name, self.flame_folder, params_path), allow_pickle=True)
        pose = torch.tensor(params['pose']).float()
        betas = torch.tensor(params['betas']).float()

        flame = {
            'shape_params': torch.cat(K * [betas[:300][None]], dim=0),
            'expression_params': torch.cat(K * [betas[300:][None]], dim=0),
            'pose_params': torch.cat(K * [torch.cat([pose[:3], pose[6:9]])[None]], dim=0),
        }

        images_list = []
        arcface_list = []

        for i in sample_list:
            image_path = images[i]
            image = np.array(imread(image_path))
            image = image / 255.
            image = image.transpose(2, 0, 1)
            arcface_image = np.load(self.get_arcface_path(image_path), allow_pickle=True)

            images_list.append(image)
            arcface_list.append(torch.tensor(arcface_image))

        images_array = torch.from_numpy(np.array(images_list)).float()
        arcface_array = torch.stack(arcface_list).float()

        return {
            'image': images_array,
            'arcface': arcface_array,
            'imagename': actor,
            'dataset': self.name,
            'flame': flame,
        }


class FlameDataset(Dataset):
    def __init__(self, name, dataset_root, num_expressions=100, num_betas=100, replica=1, use_merged_file=False):
        self.name = name
        self.dataset_root = dataset_root
        self.num_expressions = num_expressions
        self.num_betas = num_betas
        self.replica = replica
        self.use_merged_file = use_merged_file
        self.flame_folder = 'FLAME_parameters'
        self.root_path = os.path.join(self.dataset_root, self.name.upper(), self.flame_folder)

        if self.use_merged_file:
            self.merged_filepath = os.path.join(self.dataset_root, self.name.upper(), 'merged_flame_data.npz')
            merged_data = np.load(self.merged_filepath, allow_pickle=True)
            self.jaw_pose = torch.from_numpy(merged_data['jaw_pose']).float()
            self.betas = torch.from_numpy(merged_data['betas'])[:, :num_betas].float()
            self.expression = torch.from_numpy(merged_data['expression'])[:, :num_expressions].float()
        else:
            self.filenames = list(Path(self.root_path).rglob('*.npz'))
        self.base_len = len(self.expression) if self.use_merged_file else len(self.filenames)

    def __len__(self):
        return self.base_len * self.replica

    def __getitem__(self, index):
        index = index % self.base_len
        if self.use_merged_file:
            return {
                'betas': self.betas[index],
                'jaw_pose': self.jaw_pose[index],
                'expression': self.expression[index],
            }
        else:
            params = np.load(os.path.join(self.root_path, self.filenames[index]), allow_pickle=True)
            pose = torch.from_numpy(params['pose']).float()
            betas = torch.from_numpy(params['betas']).float()

            return {
                'betas': betas[:self.num_betas],    # 300, betas in FLAME
                # 'face_orient': pose[:3],    # 3, global orient in FLAME
                # 'face_pose': pose[3:],    # 12, neck, jaw, leye, reye [neck index 12 (from 0) in SMPLX]
                'jaw_pose': pose[6:9],
                'expression': betas[300:300 + self.num_expressions],
            }

