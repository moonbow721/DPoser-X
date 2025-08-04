import os
import faiss
import torch
import numpy as np

from lib.utils.metric import generate_dataset_statistics
from lib.utils.transforms import axis_angle_to_quaternion


def build_faiss_index(dataset_array, index_file_path, dataset_file_path=None):
    # Ensure the dataset tensor is a numpy array
    if isinstance(dataset_array, torch.Tensor):
        dataset_array = dataset_array.cpu().numpy()
    assert isinstance(dataset_array, np.ndarray)

    # Create a FAISS index
    faiss_dim = dataset_array.shape[-1]  # dimension of vectors
    index = faiss.index_factory(faiss_dim, "Flat")  # L2 distance (Euclidean distance)

    # Add dataset to the index
    index.add(dataset_array)

    # Save the index to a file
    faiss.write_index(index, index_file_path)
    print(f'Index saved to {index_file_path}')
    # Save the dataset to a .pt file
    if dataset_file_path is not None:
        torch.save(torch.from_numpy(dataset_array), dataset_file_path)


def prepare_body():
    data_root = './body_data'
    version = 'version1'
    args = SimpleNamespace(data_root=data_root, version=version, sample=None)
    config = SimpleNamespace(training=SimpleNamespace(batch_size=1024), eval=SimpleNamespace(batch_size=1024))

    data_module = AMASSDataModule(config, args)
    data_module.setup(stage='fit')
    dataloader = data_module.train_dataloader()

    all_poses, _, _ = generate_dataset_statistics(dataloader, key='body_pose',
                                                  output_filepath=os.path.join(data_root, version, 'statistics.npz'))
    N = all_poses.shape[0]
    all_poses = axis_angle_to_quaternion(torch.from_numpy(all_poses).reshape(N, -1, 3)).reshape(N, -1)
    build_faiss_index(all_poses, os.path.join(data_root, version, 'faiss_index_quaternion.bin'),
                      os.path.join(data_root, version, 'all_data.pt'))


def prepare_hand():
    from lib.dataset.hand import ManoDataModule
    data_root = './hand_data'
    args = SimpleNamespace(data_root=data_root, sample=None)
    config = SimpleNamespace(training=SimpleNamespace(batch_size=1024), eval=SimpleNamespace(batch_size=1024),
                             data=SimpleNamespace(dataset_names=['dex', 'freihand', 'h2o3d', 'ho3d', 'interhand26m']))
    data_module = ManoDataModule(config, args)
    data_module.setup(stage='fit')
    dataloader = data_module.train_dataloader()

    all_poses, _, _ = generate_dataset_statistics(dataloader, key='hand_pose',
                                                  output_filepath=os.path.join(data_root, 'statistics.npz'),
                                                  save_reference=True,
                                                  output_batchpath=os.path.join(data_root, 'reference_batch.pt'))
    N = all_poses.shape[0]
    all_poses = axis_angle_to_quaternion(torch.from_numpy(all_poses).cuda().reshape(N, -1, 3)).reshape(N, -1)
    build_faiss_index(all_poses, os.path.join(data_root, 'faiss_index_quaternion.bin'),
                      os.path.join(data_root, 'all_data.pt'))


def prepare_face():
    from lib.dataset.face import FlameDataModule
    data_root = './face_data'
    args = SimpleNamespace(data_root=data_root, sample=None)
    config_exp = SimpleNamespace(training=SimpleNamespace(batch_size=1024), eval=SimpleNamespace(batch_size=1024),
                                 data=SimpleNamespace(train_dataset_names=['wcpapre_train', ],
                                                      val_dataset_names=['wcpapre_valid']))
    config_shape = SimpleNamespace(training=SimpleNamespace(batch_size=1024), eval=SimpleNamespace(batch_size=1024),
                                   data=SimpleNamespace(train_dataset_names=['facewarehouse', 'florence', 'frgc', 'ft',
                                                                       'stirling', 'lyhm_train', 'wcpa_train'],
                                                        val_dataset_names=['lyhm_valid', 'wcpa_valid']))
    for config, name in zip([config_exp, config_shape], ['expression', 'shape']):
        data_module = FlameDataModule(config, args)
        data_module.setup(stage='fit')
        dataloader = data_module.train_dataloader()
        keys = ['jaw_pose', 'expression'] if name == 'expression' else 'betas'
        all_poses, _, _ = generate_dataset_statistics(dataloader, key=keys,
                                                      output_filepath=os.path.join(data_root, f'statistics_{name}.npz'),
                                                      save_reference=True,
                                                      output_batchpath=os.path.join(data_root, f'reference_batch_{name}.pt'))
        build_faiss_index(all_poses, os.path.join(data_root, f'faiss_index_{name}.bin'),
                          os.path.join(data_root, f'all_data_{name}.pt'))


def prepare_wholebody():
    from lib.dataset.whole_body import SmplxDataModule
    data_root = './wholebody_data'
    args = SimpleNamespace(data_root=data_root, sample=None)
    config = SimpleNamespace(training=SimpleNamespace(batch_size=1024), eval=SimpleNamespace(batch_size=1024),
                             data=SimpleNamespace(dataset_names=['EMAGE', 'EgoBody'],
                                                  num_expressions=100))
    data_module = SmplxDataModule(config, args)
    data_module.setup(stage='fit')
    dataloader = data_module.train_dataloader()

    data_root = './wholebody_data_tmp'
    keys = ['body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression']
    all_poses, _, _ = generate_dataset_statistics(dataloader, key=keys,
                                                  output_filepath=os.path.join(data_root, 'statistics.npz'),
                                                  save_reference=True,
                                                  output_batchpath=os.path.join(data_root, 'reference_batch.pt'))
    build_faiss_index(all_poses, os.path.join(data_root, 'faiss_index.bin'),
                      os.path.join(data_root, 'all_data.pt'))


if __name__ == '__main__':
    from lib.dataset.body import AMASSDataModule
    from types import SimpleNamespace
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')

    prepare_body()
    prepare_hand()
    prepare_face()
    prepare_wholebody()
