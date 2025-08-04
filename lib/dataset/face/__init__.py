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
import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
from ml_collections import ConfigDict
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from lib.dataset.face.base import BaseDataset, FlameDataset


N_POSES = 3 + 100


def build_train(config, device):
    data_list = []
    total_images = 0
    for dataset in config.training_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        dataset = BaseDataset(name=dataset_name, config=config, device=device, isEval=False)
        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images


def build_val(config, device):
    data_list = []
    total_images = 0
    for dataset in config.eval_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        dataset = BaseDataset(name=dataset_name, config=config, device=device, isEval=True)
        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images


class FlameDataModule(pl.LightningDataModule):
    def __init__(self, config: ConfigDict, args) -> None:
        """
        Initialize LightningDataModule for hand prior training
        """
        super().__init__()
        self.config = config
        self.args = args

    def setup(self, stage=None):
        support_train_datasets = ['FACEWAREHOUSE', 'FLORENCE', 'FRGC', 'FT', 'STIRLING',
                                  'LYHM_TRAIN', 'WCPA_TRAIN', 'WCPAPRE_TRAIN']
        support_val_datasets = ['LYHM_VALID', 'WCPA_VALID', 'WCPAPRE_VALID']
        train_dataset_names = self.config.data.train_dataset_names
        val_dataset_names = self.config.data.val_dataset_names
        replica = getattr(self.config.data, 'replica', 1)
        num_expressions = getattr(self.config.data, 'num_expressions', 100)
        num_betas = getattr(self.config.data, 'num_betas', 100)
        print(f'setup with replica={replica}, num_expressions={num_expressions}, num_betas={num_betas}')
        assert all(dataset_name.upper() in support_train_datasets for dataset_name in
                   train_dataset_names), "Some train dataset names are not in the expected list."
        assert all(dataset_name.upper() in support_val_datasets for dataset_name in
                   val_dataset_names), "Some val dataset names are not in the expected list."

        train_datasets = {dataset: FlameDataset(dataset, self.args.data_root, num_expressions, num_betas,
                                                replica=replica, use_merged_file=True) for dataset in train_dataset_names}
        val_datasets = {dataset: FlameDataset(dataset, self.args.data_root, num_expressions, num_betas,
                                              replica=replica, use_merged_file=True) for dataset in val_dataset_names}
        print(f'Loaded {len(train_datasets)} training datasets and {len(val_datasets)} validation datasets.')

        if stage == 'fit' or stage is None:
            self.train_dataset = ConcatDataset(list(train_datasets.values()))
            self.val_dataset = ConcatDataset(list(val_datasets.values()))

            # Balance samples in training dataset
            # dataset_weights = {'FACEWAREHOUSE': 20.0, 'FLORENCE': 60.0, 'FRGC': 5.0, 'FT': 30.0,
            #                    'STIRLING': 20.0, 'LYHM_TRAIN': 3.0, 'WCPA_TRAIN': 15.0, 'WCPAPRE_TRAIN': 1.0}
            dataset_weights = {'FACEWAREHOUSE': 1.0, 'FLORENCE': 1.0, 'FRGC': 1.0, 'FT': 1.0,
                               'STIRLING': 1.0, 'LYHM_TRAIN': 1.0, 'WCPA_TRAIN': 1.0, 'WCPAPRE_TRAIN': 1.0}

            sample_weights = []
            for dataset_name, dataset in train_datasets.items():
                weight = dataset_weights[dataset_name.upper()]
                sample_weights += [weight] * len(dataset)

            self.sample_weights = torch.tensor(sample_weights)
            self.sampler = WeightedRandomSampler(self.sample_weights, len(self.sample_weights))

        if stage == 'test' or stage is None:
            self.test_dataset = ConcatDataset(list(val_datasets.values()))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.training.batch_size,
                          num_workers=8, shuffle=False, drop_last=True, sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.eval.batch_size,
                          num_workers=8, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.eval.batch_size,
                          num_workers=8, shuffle=False, drop_last=True)


def get_args():
    parser = argparse.ArgumentParser(description='tmp')
    parser.add_argument('--data-root', type=str,
                        default='/data3/ljz24/projects/3d/DPoser/face_data', help='dataset root')

    return parser.parse_args()

