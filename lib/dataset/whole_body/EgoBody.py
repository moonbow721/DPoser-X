import os

import torch
from torch.utils.data import Dataset


class EgoBodyDataset(Dataset):
    def __init__(self, name, dataset_root, num_expressions=100, sample_interval=None):
        self.name = name
        self.dataset_root = dataset_root
        self.num_expressions = num_expressions
        self.param_file_path = os.path.join(self.dataset_root, f'{self.name}.pt')

        # Load the dataset
        if os.path.exists(self.param_file_path):
            self.data = torch.load(self.param_file_path)
        else:
            raise FileNotFoundError(f"No processed file found at {self.param_file_path}")

        if sample_interval:
            self._sample(sample_interval)
        self.dataset_size = self.data['global_orient'].shape[0]

    def _sample(self, sample_interval):
        print(f'Class EgoBodyDataset({self.name}): sample dataset every {sample_interval} frame')
        for key in self.data.keys():
            self.data[key] = self.data[key][::sample_interval]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # Extract the item for each key at the given index
        sample = {
            'global_orient': self.data['global_orient'][index],
            'body_pose': self.data['body_pose'][index],
            'left_hand_pose': self.data['left_hand_pose'][index],
            'right_hand_pose': self.data['right_hand_pose'][index],
            'jaw_pose': self.data['jaw_pose'][index],
            'expression': self.data['expression'][index, :self.num_expressions],
        }
        return sample

