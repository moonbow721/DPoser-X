import json
import os
import warnings
from PIL import Image
from pathlib import Path

import cv2
import numpy as np
import torch

from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data._utils.collate import default_collate

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from lib.body_model.joint_mapping import vitpose_to_openpose, get_openpose_part
from lib.utils.preprocess import crop_img_tensor
from lib.utils.misc import to_device


class WcpapreDataset(Dataset):
    def __init__(self, image_root, flame_root, deca_root=None, select_file=None, cropped_size=256, image_size=256,
                 sample=1, is_crop=False, num_expressions=100, num_betas=100):
        self.image_root = image_root
        self.flame_root = flame_root
        self.deca_root = deca_root
        self.cropped_size = cropped_size
        self.image_size = image_size
        self.sample = sample
        self.num_expressions = num_expressions
        self.num_betas = num_betas

        self.flame_folder = 'FLAME_parameters'
        self.flame_root_path = os.path.join(self.flame_root, self.flame_folder)

        self.image_folder = 'arcface_input' if is_crop else 'images'
        self.image_suffix = 'png' if is_crop else 'jpg'
        self.image_root_path = os.path.join(self.image_root, self.image_folder)

        self.keypoint_root = self.image_root_path + '_kpts'

        self.flame_filenames = sorted(list(Path(self.flame_root_path).rglob('*.npz')))
        self.image_filenames = sorted(list(Path(self.image_root_path).rglob(f'*.{self.image_suffix}')))
        image_basenames = [os.path.relpath(f, self.image_root_path) for f in self.image_filenames]
        self.kpt_filenames = [os.path.join(self.keypoint_root, f.replace(f'.{self.image_suffix}', '.json'))
                              for f in image_basenames]

        # Select a subset of files based on the provided list
        if select_file is not None:
            with open(select_file, 'r') as f:
                select_list = f.readlines()
            select_uids = sorted([self.get_uid(x) for x in select_list])
        else:
            select_uids = None

        # Initialize lists to track missing files across different data types
        missing_uids = set()

        # Check for missing DECA files if DECA root is specified
        if self.deca_root:
            self.deca_filenames = [os.path.join(self.deca_root, f.replace(f'.{self.image_suffix}', '.npz')) for f in
                                   image_basenames]
            missing_uids.update(self.get_uid(f) for f in self.deca_filenames if not os.path.exists(f))

        # Update for missing keypoint files
        missing_uids.update(self.get_uid(f) for f in self.kpt_filenames if not os.path.exists(f))

        # Update for missing image files based on flame filenames
        base_flame_uids = set(self.get_uid(f) for f in self.flame_filenames)
        base_image_uids = set(self.get_uid(f) for f in self.image_filenames)
        missing_uids.update(base_flame_uids.difference(base_image_uids))

        # Filter out all missing UIDs from all file lists
        if missing_uids:
            warnings.warn(f"Missing files: {sorted(missing_uids)}")
            self.image_filenames = sorted(f for f in self.image_filenames if self.get_uid(f) not in missing_uids)
            self.kpt_filenames = sorted(f for f in self.kpt_filenames if self.get_uid(f) not in missing_uids)
            self.flame_filenames = sorted(f for f in self.flame_filenames if self.get_uid(f) not in missing_uids)
            if self.deca_root:
                self.deca_filenames = sorted(f for f in self.deca_filenames if self.get_uid(f) not in missing_uids)

        if select_uids:
            self.image_filenames = [f for f in self.image_filenames if self.get_uid(f) in select_uids]
            self.kpt_filenames = [f for f in self.kpt_filenames if self.get_uid(f) in select_uids]
            self.flame_filenames = [f for f in self.flame_filenames if self.get_uid(f) in select_uids]
            if self.deca_root:
                self.deca_filenames = [f for f in self.deca_filenames if self.get_uid(f) in select_uids]

        assert len(self.flame_filenames) == len(self.image_filenames) == len(self.kpt_filenames), \
            f"Number of FLAME parameters, images, and keypoints do not match: {len(self.flame_filenames)}, {len(self.image_filenames)}, {len(self.kpt_filenames)}"
        if self.deca_root is not None:
            assert len(self.flame_filenames) == len(self.deca_filenames), \
                f"Number of FLAME parameters and DECA parameters do not match: {len(self.flame_filenames)}, {len(self.deca_filenames)}"

        self.base_len = len(self.flame_filenames)
        self.face_idx = get_openpose_part('face')

        # Apply sampling
        if self.sample != 1:
            sampled_indices = np.arange(0, self.base_len, step=self.sample)
            self.flame_filenames = [self.flame_filenames[i] for i in sampled_indices]
            self.image_filenames = [self.image_filenames[i] for i in sampled_indices]
            self.kpt_filenames = [self.kpt_filenames[i] for i in sampled_indices]
            if self.deca_root:
                self.deca_filenames = [self.deca_filenames[i] for i in sampled_indices]
            self.base_len = len(self.flame_filenames)

    def get_uid(self, full_path):
        # Example: ./root_path/1606018605840_ar.jpg -> 1606018605840
        return os.path.basename(full_path).split('_')[0]

    def __len__(self):
        return self.base_len

    def __getitem__(self, index):
        # check the uid is the same
        assert self.get_uid(self.image_filenames[index]) == self.get_uid(self.flame_filenames[index]) == self.get_uid(self.kpt_filenames[index]), \
            f"UID mismatch: {self.get_uid(self.image_filenames[index])}, {self.get_uid(self.flame_filenames[index])}, {self.get_uid(self.kpt_filenames[index])}"
        uid = self.get_uid(self.flame_filenames[index])

        # Load GT FLAME parameters
        params = np.load(os.path.join(self.flame_filenames[index]), allow_pickle=True)
        trans = torch.from_numpy(params['trans']).float()
        pose = torch.from_numpy(params['pose']).float()
        betas = torch.from_numpy(params['betas']).float()

        # Load keypoints
        with open(self.kpt_filenames[index], 'r') as file:
            json_data = json.load(file)
        kpts = np.array(json_data[0]['keypoints'])
        kpts = vitpose_to_openpose(kpts)  # [133, 3] -> [135, 3]
        kpts = kpts[self.face_idx]  # [135, 3] -> [68, 3]

        # Load image
        image = Image.open(self.image_filenames[index]).convert('RGB')
        image_tensor = ToTensor()(image)

        # Compute bounding box from keypoints
        bbox = [np.min(kpts[:, 0]), np.min(kpts[:, 1]), np.max(kpts[:, 0]), np.max(kpts[:, 1])]

        # Crop and process the image
        crop_image_tensor, new_bbox, new_kpts = crop_img_tensor(image_tensor, bbox, self.cropped_size, kpts)

        # Normalize keypoints to [-1, 1]
        landmark = new_kpts.copy()
        landmark[:, 0] = landmark[:, 0] / float(crop_image_tensor.shape[2]) * 2 - 1
        landmark[:, 1] = landmark[:, 1] / float(crop_image_tensor.shape[1]) * 2 - 1

        # Resize the image
        crop_image_tensor = interpolate(crop_image_tensor.unsqueeze(0), size=(self.image_size, self.image_size),
                                        mode='bilinear', align_corners=False)[0]
        inputs = {
            'uid': uid,
            'original_image': image_tensor,  # Original full image tensor
            'image': crop_image_tensor,  # Cropped and processed image tensor
            'original_kpts': torch.from_numpy(kpts),  # Original keypoints
            'kpts': torch.from_numpy(landmark),
        }
        if self.deca_root:
            deca_params = np.load(self.deca_filenames[index], allow_pickle=True)
            # DECA/EMOCA parameters, {'pose': 6-dim, 'shape': 100-dim, 'exp': 50-dim, 'cam': 3-dim}
            inputs['deca_params'] = deca_params
        gt_flame = {
            'betas': betas[:self.num_betas],
            'trans': trans,
            'face_orient': pose[:3],
            'face_pose': pose[3:],  # 12, neck, jaw, leye, reye [not used]
            'jaw_pose': pose[6:9],
            'expression': betas[300:300 + self.num_expressions],
        }
        return {**inputs, **gt_flame}



class FaceParser:
    def __init__(self, device='cuda', mini_batchsize=5):
        self.image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.seg_net = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        self.device = device
        self.move_to_device(device)
        self.mini_batchsize = mini_batchsize

    def move_to_device(self, device):
        self.seg_net = self.seg_net.to(device)
        self.device = device

    def get_image_mask(self, image_batch):
        """
        Processes a batch of images by dividing it into smaller mini-batches.

        Args:
            image_batch (torch.Tensor): shape (batch_size, 3, height, width)

        Returns:
            mask_resized (torch.Tensor): shape (batch_size, 1, height, width)
        """
        n = len(image_batch)
        all_masks = []

        for start_idx in range(0, n, self.mini_batchsize):
            end_idx = start_idx + self.mini_batchsize
            mini_batch = image_batch[start_idx:end_idx]
            mask_resized = self.process_mini_batch(mini_batch)
            all_masks.append(mask_resized)

        # Concatenate all mini-batch masks back into a single batch
        full_mask = torch.cat(all_masks, dim=0)
        return full_mask

    def process_mini_batch(self, mini_batch):
        """
        Processes a single mini-batch of images.

        Args:
            mini_batch (torch.Tensor): shape (mini_batch_size, 3, height, width)

        Returns:
            mask_resized (torch.Tensor): shape (mini_batch_size, 1, height, width)
        """
        inputs = self.image_processor(images=mini_batch, return_tensors="pt", do_rescale=False).to(self.device)
        outputs = self.seg_net(**inputs)
        logits = outputs.logits  # shape: (mini_batch_size, num_labels, height/4, width/4)

        # Get the predicted classes for each pixel
        predicted_classes = torch.argmax(logits, dim=1, keepdim=True)  # shape: (mini_batch_size, 1, height/4, width/4)

        # Define face areas (semantic labels corresponding to face parts)
        face_area = torch.tensor([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17]).to(predicted_classes.device)

        # Create masks for face areas
        mask = torch.zeros_like(predicted_classes, dtype=torch.float32)
        for category in face_area:
            mask += (predicted_classes == category).float()

        # Resize masks to the original image size
        original_size = mini_batch.shape[2:]  # Original size is (height, width)
        mask_resized = interpolate(mask, size=original_size, mode='nearest')

        return mask_resized


def wcpapre_collate(batch):
    batch_mod = [{k: v for k, v in item.items() if k != 'original_image'} for item in batch]
    collated_data = default_collate(batch_mod)

    # Handle the original_image separately
    original_images = [item['original_image'] for item in batch]
    collated_data['original_image'] = original_images  # Store as a list or any other suitable format

    return collated_data
