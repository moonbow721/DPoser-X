import argparse
import os
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from ml_collections import ConfigDict
from torch.utils.data import DataLoader, ConcatDataset, random_split

from .dataset import Dataset
from .mocap_dataset import MoCapDataset, InterHandDataset, ManoDataset

# Local poses without global orient
N_POSES = 15


class ManoDataModule(pl.LightningDataModule):
    def __init__(self, config: ConfigDict, args) -> None:
        """
        Initialize LightningDataModule for hand prior training
        """
        super().__init__()
        self.config = config
        self.args = args

    def setup(self, stage=None):
        # Prepare data for training
        expected_datasets = ['dex', 'freihand', 'h2o3d', 'ho3d', 'interhand26m']
        dataset_names = self.config.data.dataset_names
        assert all(dataset_name in expected_datasets for dataset_name in
                   dataset_names), "Some dataset names are not in the expected list."
        # mocap_dataset = MoCapDataset(os.path.join(self.args.data_root, 'freihand_mocap.npz'))
        image_mano_datasets = [ManoDataset(os.path.join(self.args.data_root, 'dataset_params', f'{dataset}.npz'))
                               for dataset in dataset_names]
        reinterhand_dataset = InterHandDataset(os.path.join(self.args.data_root, 'reinterhand_mocap.pt'),
                                               single_hand=True)

        # Spliting the whole reinterhand dataset
        torch.manual_seed(0)
        total_size = len(reinterhand_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(reinterhand_dataset, [train_size, val_size, test_size])

        if stage == 'fit' or stage is None:
            self.train_dataset = ConcatDataset([train_dataset, *image_mano_datasets])
            self.sample_weights = torch.ones(len(self.train_dataset))
            self.val_dataset = val_dataset

        # Prepare data for testing
        if stage == 'test' or stage is None:
            self.test_dataset = test_dataset


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.training.batch_size,
                          num_workers=8, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.eval.batch_size,
                          num_workers=8, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.eval.batch_size,
                          num_workers=8, shuffle=False, drop_last=True)


