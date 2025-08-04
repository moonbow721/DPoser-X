import os
import copy
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import pytorch_lightning as pl

# For lightning training
from lib.dataset.whole_body.EgoBody import EgoBodyDataset
from lib.dataset.whole_body.GRAB import GRABDataset
from lib.dataset.whole_body.EMAGE import EMAGEDataset
from lib.dataset.whole_body.Arctic import ArcticDataset

from lib.dataset.body import AMASSDataModule
from lib.dataset.hand import ManoDataModule, InterHandDataset
from lib.dataset.face import FlameDataModule

POSES_LIST = [21, 15, 100 + 3]


class SmplxDataModule(pl.LightningDataModule):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args

    def setup(self, stage=None):
        # Prepare data for training and validation
        dataset_names = self.config.data.dataset_names
        train_datasets, val_datasets, test_datasets = {}, {}, {}
        test_sample_interval = getattr(self.args, 'sample', 10)
        # used FLAME model and face-only dataset: 100 expression
        # TCDHands [in AMASS]: no jaw, no eyes, no expression

        # Arctic: no jaw, no eyes, no expression, two hands interaction
        # GRAB [in AMASS]: has jaw (bad quality), no eyes, 80 expression, grabbing objects
        # EgoBody: has jaw, has eyes, 10 expression, two-people interaction, standing/sitting 【with gender】
        # EMAGE/BEAT: has jaw, has eyes, 100 expression, 30fps, standing and talking

        if 'EgoBody' in dataset_names:
            for dataset_name in ['smplx_camera_wearer', 'smplx_interactee']:
                for dataset_type in ['train', 'val', 'test']:
                    dataset = EgoBodyDataset(name=f'{dataset_name}_{dataset_type}',
                                             dataset_root=os.path.join(self.args.data_root, 'EgoBody',
                                                                       'smplx_parameters'),
                                             num_expressions=self.config.data.num_expressions,
                                             sample_interval=None if dataset_type == 'train' else test_sample_interval)
                    if dataset_type == 'train':
                        train_datasets[f'{dataset_name}_{dataset_type}'] = WholebodyWrapper(dataset)
                        print(f"Loaded {len(dataset)} wholebody samples from {dataset_name}_{dataset_type}")
                    elif dataset_type == 'val':
                        val_datasets[f'{dataset_name}_{dataset_type}'] = dataset
                    else:
                        test_datasets[f'{dataset_name}_{dataset_type}'] = dataset

        if 'GRAB' in dataset_names:
            dataset = GRABDataset(dataset_root=os.path.join(self.args.data_root, 'GRAB'), split='train', mask_face=True)
            train_datasets['GRAB'] = WholebodyWrapper(dataset)
            print(f"Loaded {len(dataset)} wholebody samples from GRAB")

        if 'EMAGE' in dataset_names:
            dataset = EMAGEDataset(dataset_root=os.path.join(self.args.data_root, 'BEAT'), split='train')
            train_datasets['EMAGE'] = WholebodyWrapper(dataset)
            print(f"Loaded {len(dataset)} wholebody samples from EMAGE")

        if 'Arctic' in dataset_names:
            dataset = ArcticDataset(dataset_root=os.path.join(self.args.data_root, 'Arctic', 'merged_smplx'), split='train', mask_face=True)
            train_datasets['Arctic'] = WholebodyWrapper(dataset)
            print(f"Loaded {len(dataset)} wholebody samples from Arctic")

        if stage == 'fit' or stage is None:
            self.train_dataset = ConcatDataset(list(train_datasets.values()))
            self.val_dataset = ConcatDataset(list(val_datasets.values()))

        if stage == 'test' or stage is None:
            self.test_dataset = ConcatDataset(list(test_datasets.values()))

        dataset_weights = {'EgoBody_smplx_camera_wearer_train': 5.0,
                           'EgoBody_smplx_interactee_train': 5.0,
                           'GRAB': 1.0,
                           'EMAGE': 0.5,
                           'Arctic': 5.0}

        sample_weights = []
        for dataset_name, dataset in train_datasets.items():
            weight = dataset_weights.get(dataset_name, 1.0)
            sample_weights += [weight] * len(dataset)

        self.sample_weights = torch.tensor(sample_weights)
        self.sampler = WeightedRandomSampler(self.sample_weights, len(self.sample_weights))

    # FIXME: there seems to be a bug caused by WholebodyWrapper for num_workers > 0
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.training.batch_size,
                          num_workers=0, drop_last=True, sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.eval.batch_size,
                          num_workers=0, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.eval.batch_size,
                          num_workers=0, shuffle=False, drop_last=True)


class SmplxMixedDataModule(pl.LightningDataModule):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        root_dir = os.path.dirname(args.data_root)

        # wholebody
        self.smplx_data_module = SmplxDataModule(config, args)
        self.smplx_weight = 1.0

        # body only
        body_args = copy.deepcopy(args)
        body_args.data_root = os.path.join(root_dir, 'body_data')
        body_args.version = config.data.amass_version
        body_args.sample = config.data.amass_sample
        self.body_data_module = AMASSDataModule(config, body_args)
        self.body_weight = config.data.amass_weight

        # single hand & two hands
        hand_args = copy.deepcopy(args)
        hand_args.data_root = os.path.join(root_dir, 'hand_data')
        hand_config = copy.deepcopy(config)
        hand_config.data.dataset_names = config.data.hamer_dataset_names
        self.hand_data_module = ManoDataModule(hand_config, hand_args)
        self.hand_weight = config.data.hamer_weight
        self.twohand_dataset = InterHandDataset(os.path.join(hand_args.data_root, 'reinterhand_mocap.pt'),
                                                single_hand=False)
        self.twohand_sample_weights = torch.ones(len(self.twohand_dataset))
        self.twohand_weight = config.data.interhand_weight

        # face only
        face_args = copy.deepcopy(args)
        face_args.data_root = os.path.join(root_dir, 'face_data')
        face_config = copy.deepcopy(config)
        face_config.data.train_dataset_names = config.data.face_train_dataset_names
        face_config.data.val_dataset_names = config.data.face_val_dataset_names
        self.face_data_module = FlameDataModule(face_config, face_args)
        self.face_weight = config.data.face_weight

    def setup(self, stage=None):
        # Setup for each data module
        self.smplx_data_module.setup(stage)
        self.body_data_module.setup(stage)
        self.hand_data_module.setup(stage)
        self.face_data_module.setup(stage)

        # wrap datasets to unify the output format
        smplx_train_dataset = WholebodyWrapper(self.smplx_data_module.train_dataset, part_type='wholebody')
        body_train_dataset = WholebodyWrapper(self.body_data_module.train_dataset, part_type='body_only')
        hand_train_dataset = WholebodyWrapper(self.hand_data_module.train_dataset, part_type='hand_only')
        twohand_train_dataset = WholebodyWrapper(self.twohand_dataset, part_type='two_hands')
        face_train_dataset = WholebodyWrapper(self.face_data_module.train_dataset, part_type='face_only')

        # Combine train datasets
        self.train_dataset = ConcatDataset([
            smplx_train_dataset, body_train_dataset, hand_train_dataset,
            twohand_train_dataset, face_train_dataset
        ])

        # Get and adjust sample weights
        smplx_weights = self.smplx_data_module.sample_weights * self.smplx_weight
        body_weights = self.body_data_module.sample_weights * self.body_weight
        hand_weights = self.hand_data_module.sample_weights * self.hand_weight
        twohand_weights = self.twohand_sample_weights * self.twohand_weight
        face_weights = self.face_data_module.sample_weights * self.face_weight

        print(f"Combined train dataset samples with \n"
              f"wholebody: {len(self.smplx_data_module.train_dataset)}, weight: {self.smplx_weight} \n"
              f"body: {len(self.body_data_module.train_dataset)}, weight: {self.body_weight}, \n"
              f"hand: {len(self.hand_data_module.train_dataset)}, weight: {self.hand_weight}, \n"
              f"twohand: {len(self.twohand_dataset)}, weight: {self.twohand_weight}, \n"
              f"face: {len(self.face_data_module.train_dataset)}, weight: {self.face_weight} \n")

        # Combine weights
        combined_weights = torch.cat((smplx_weights, body_weights, hand_weights, twohand_weights, face_weights), dim=0)

        # Create new sampler
        self.sampler = WeightedRandomSampler(combined_weights, len(combined_weights))

    # FIXME: there seems to be a bug caused by WholebodyWrapper for num_workers > 0
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.training.batch_size,
                          num_workers=0, drop_last=True, sampler=self.sampler)

    def val_dataloader(self):
        return self.smplx_data_module.val_dataloader()

    def test_dataloader(self):
        return self.smplx_data_module.test_dataloader()


class WholebodyWrapper(Dataset):
    def __init__(self, dataset, part_type='wholebody', body_pose_dim=63, hand_pose_dim=45,
                 jaw_pose_dim=3, num_expressions=100):
        self.dataset = dataset
        self.part_type = part_type
        assert part_type in ['wholebody', 'two_hands', 'body_only', 'hand_only', 'face_only'], 'Invalid part type'
        self.body_pose_dim = body_pose_dim
        self.hand_pose_dim = hand_pose_dim
        self.jaw_pose_dim = jaw_pose_dim
        self.num_expressions = num_expressions
        self.necessary_keys = {'body_pose': body_pose_dim, 'left_hand_pose': hand_pose_dim,
                               'right_hand_pose': hand_pose_dim, 'jaw_pose': jaw_pose_dim,
                               'expression': num_expressions}
        if part_type == 'wholebody':
            self.mask = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
        elif part_type == 'two_hands':
            self.mask = torch.tensor([0, 1, 1, 0], dtype=torch.bool)
        elif part_type == 'body_only':
            self.mask = torch.tensor([1, 0, 0, 0], dtype=torch.bool)
        elif part_type == 'face_only':
            self.mask = torch.tensor([0, 0, 0, 1], dtype=torch.bool)
        elif part_type == 'hand_only':
            self.mask = None
        else:
            raise ValueError(f'Invalid part type: {part_type}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.part_type == 'hand_only':
            return self.getitem_single_hand(index)

        available_sample = self.dataset[index]
        # Initialize the complete sample with zeros
        complete_sample = {}
        for key, dim in self.necessary_keys.items():
            complete_sample[key] = available_sample.get(key, torch.zeros(dim))
        complete_sample['mask'] = available_sample.get('mask', self.mask.clone())

        return complete_sample

    def getitem_single_hand(self, index):
        available_sample = self.dataset[index]
        right_hand = index % 2 == 0  # True for right hand, False for left hand
        complete_sample = {}
        for key, dim in self.necessary_keys.items():
            if key == 'right_hand_pose' and right_hand or key == 'left_hand_pose' and not right_hand:
                fetch_key = 'hand_pose'
            else:
                fetch_key = key
            complete_sample[key] = available_sample.get(fetch_key, torch.zeros(dim))
        complete_sample['mask'] = torch.tensor([0, 0, 1, 0], dtype=torch.bool) if right_hand \
            else torch.tensor([0, 1, 0, 0], dtype=torch.bool)

        return complete_sample

