import os
import numpy as np
from pathlib import Path

def merge_flame_data(dataset_root, name, num_expressions=100):
    flame_folder = 'FLAME_parameters'
    save_path = os.path.join(dataset_root, name.upper())
    root_path = os.path.join(dataset_root, name.upper(), flame_folder)
    filenames = list(Path(root_path).rglob('*.npz'))

    jaw_pose_list = []
    expression_list = []
    betas_list = []

    for filename in filenames:
        params = np.load(os.path.join(root_path, filename), allow_pickle=True)
        pose = params['pose']
        betas = params['betas']

        jaw_pose = pose[6:9]  # 3, jaw pose
        shape = betas[:300]  # 300, shape
        expression = betas[300:300 + num_expressions]  # 100, expression

        jaw_pose_list.append(jaw_pose)
        betas_list.append(shape)
        expression_list.append(expression)

    jaw_poses = np.array(jaw_pose_list)
    betas = np.array(betas_list)
    expressions = np.array(expression_list)

    np.savez(os.path.join(save_path, 'merged_flame_data.npz'),
             jaw_pose=jaw_poses, expression=expressions, betas=betas)
    print(f'Merged data saved to {os.path.join(save_path, "merged_flame_data.npz")}')
    

if __name__ == '__main__':
    merge_flame_data('./data/face_data', 'FACEWAREHOUSE')

