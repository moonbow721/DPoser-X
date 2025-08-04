import os
import pickle
import numpy as np


def process_hand_pose(hand_pose, right):
    """
    Process the hand_pose array based on the 'right' flag.
    """
    if not right:
        # print('flipping hand_pose')
        hand_pose[1::3] *= -1
        hand_pose[2::3] *= -1
    return hand_pose


def collect_data(directory):
    """
    Collects hand_pose and betas data from .data.pyd files in the given directory.
    """
    hand_poses = []
    betas_array = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.data.pyd'):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)[0]

                if data['has_hand_pose'] and data['has_betas']:
                    # print('valid data')
                    hand_pose = process_hand_pose(data['hand_pose'], data['right'])
                    hand_poses.append(hand_pose)
                    betas_array.append(data['betas'])

    return hand_poses, betas_array


def save_to_npz(hand_poses, betas_array, output_file):
    """
    Save the hand_pose and betas data to an npz file.
    """
    np.savez(output_file, hand_pose=np.array(hand_poses), betas=np.array(betas_array))


if __name__ == '__main__':
    root_path = '/data3/ljz24/projects/3d/data/human/Handdataset/HaMeR/hamer_training_data'
    input_root = os.path.join(root_path, 'dataset_contents')
    output_root = os.path.join(root_path, 'dataset_params')
    for dataset in os.listdir(input_root):
        print(f"Processing {dataset}")
        # Directory containing the .data.pyd files
        directory = os.path.join(input_root, dataset)
        dataset_name = dataset.split('-')[0]

        # Collect data
        hand_poses, betas_array = collect_data(directory)

        # Save to npz file
        output_file = os.path.join(output_root, f'{dataset_name}.npz')
        save_to_npz(hand_poses, betas_array, output_file)
        print(f"Data saved to {output_file}")
