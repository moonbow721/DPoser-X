import os

import torch
from torch.utils.data import Dataset


class GRABDataset(Dataset):
    def __init__(self, dataset_root, split='train', num_expressions=100, mask_face=False, sample_interval=None):
        """
        Args:
            dataset_root (str): The root directory where the dataset is stored.
            split (str): The dataset split, e.g., 'train', 'val', 'test'.
            num_expressions (int): The number of expression coefficients to use.
            mask_face (bool): Whether to mask the face in the dataset. [due to low quality of face data]
        """
        self.dataset_root = os.path.join(dataset_root, split)
        self.num_expressions = num_expressions

        # Load the dataset
        self.data = {}
        self.load_dataset()

        # Use the first key to determine the dataset size
        self.dataset_size = next(iter(self.data.values())).shape[0]
        # Create a mask for the dataset
        if mask_face:
            self.data['mask'] = self.create_mask()
        if sample_interval:
            self._sample(sample_interval)
            self.dataset_size = next(iter(self.data.values())).shape[0]

    def load_dataset(self):
        """Loads the dataset parameters from .pt files."""
        parameters = ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression']
        for param in parameters:
            file_path = os.path.join(self.dataset_root, f'{param}.pt')
            if os.path.exists(file_path):
                self.data[param] = torch.load(file_path)
            else:
                raise FileNotFoundError(f"No file found for {param} at {file_path}")

    def create_mask(self):
        """Creates a mask for the dataset."""
        # mask: [dataset_size, 4], 0 for body, 1 for left_hand, 2 for right_hand, 3 for face
        mask_type = torch.tensor([1, 1, 1, 0], dtype=torch.bool)  # mask the face
        mask = mask_type.repeat(self.dataset_size, 1)
        return mask

    def _sample(self, sample_interval):
        print(f'Sample dataset {self.dataset_root} every {sample_interval} frame')
        for key in self.data.keys():
            self.data[key] = self.data[key][::sample_interval]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # Extract the item for each key at the given index, with special handling for 'expression'
        sample = {param: self.data[param][index] for param in self.data}
        sample['expression'] = sample['expression'][:self.num_expressions]
        return sample


