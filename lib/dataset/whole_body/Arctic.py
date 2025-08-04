import os

import torch

from lib.dataset.whole_body.GRAB import GRABDataset


class ArcticDataset(GRABDataset):
    def __init__(self, dataset_root, split='train', num_expressions=100, mask_face=False, sample_interval=None):
        """
        Args:
            dataset_root (str): The root directory where the dataset is stored.
            split (str): The dataset split, e.g., 'train', 'val', 'test'.
            num_expressions (int): The number of expression coefficients to use.
            mask_face (bool): Whether to mask the face in the dataset. [due to low quality of face data]
        """
        # Call the constructor of the parent class
        super().__init__(dataset_root, split, num_expressions, mask_face, sample_interval)

    def load_dataset(self):
        """Loads a subset of dataset parameters from .pt files."""
        parameters = ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose']
        for param in parameters:
            file_path = os.path.join(self.dataset_root, f'{param}.pt')
            if os.path.exists(file_path):
                self.data[param] = torch.load(file_path)
            else:
                raise FileNotFoundError(f"No file found for {param} at {file_path}")
        self.data['jaw_pose'] = torch.zeros(len(self.data['global_orient']), 3)
        self.data['expression'] = torch.zeros(len(self.data['global_orient']), self.num_expressions)

