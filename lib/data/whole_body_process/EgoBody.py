import glob
import os
import pickle
import re

import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from smplx import MANO as orig_MANO

from lib.body_model.hand_model import MANO

left_hand_model = orig_MANO(model_path='/data3/ljz24/projects/3d/body_models/mano',
                            is_rhand=False, batch_size=50, num_pca_comps=12, )
right_hand_model = orig_MANO(model_path='/data3/ljz24/projects/3d/body_models/mano',
                             is_rhand=True, batch_size=50, num_pca_comps=12, )
left_pca_comps = left_hand_model.hand_components
right_pca_comps = right_hand_model.hand_components


def debug():
    file_path = '/data3/ljz24/projects/3d/data/human/WholeBodydataset/EgoBody/smplx_parameters/smplx_camera_wearer_val/recording_20210921_S11_S10_02/body_idx_1/results/frame_02192/000.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    right_hand_pose = torch.from_numpy(data['right_hand_pose'])
    print(right_hand_pose.shape)
    right_hand_pose = torch.einsum('bi,ij->bj', [right_hand_pose, right_pca_comps])
    print(right_hand_pose.shape)

    model = MANO(model_path='/data3/ljz24/projects/3d/body_models/mano', is_rhand=True,
                 batch_size=1)
    output = model(hand_pose=right_hand_pose)
    from lib.body_model.visual import render_mesh
    import cv2

    bg_img = np.ones([192, 256, 3]) * 255  # background canvas
    focal = [5000, 5000]
    princpt = [80, 128]

    mesh = output.v.detach().cpu().numpy()[0]
    rendered_img = render_mesh(bg_img, mesh, model.faces, {'focal': focal, 'princpt': princpt},
                               view='half_right_bottom')
    cv2.imwrite('pca_debug.png', rendered_img)


def process_sample(input_dict, num_expression=100):
    output_dict = {}
    left_hand_pose = torch.from_numpy(input_dict['left_hand_pose'])
    right_hand_pose = torch.from_numpy(input_dict['right_hand_pose'])
    left_hand_pose = torch.einsum('bi,ij->bj', [left_hand_pose, left_pca_comps])
    right_hand_pose = torch.einsum('bi,ij->bj', [right_hand_pose, right_pca_comps])

    output_dict['global_orient'] = torch.from_numpy(input_dict['global_orient'])
    output_dict['body_pose'] = torch.from_numpy(input_dict['body_pose'])
    output_dict['left_hand_pose'] = left_hand_pose
    output_dict['right_hand_pose'] = right_hand_pose
    output_dict['jaw_pose'] = torch.from_numpy(input_dict['jaw_pose'])

    # Retrieve the 'expression' tensor
    expression = torch.from_numpy(input_dict['expression'])
    # Calculate the padding size needed to reach num_expression
    current_size = expression.size(1)
    padding_size = num_expression - current_size
    # Apply zero-padding to the 'expression' tensor
    if padding_size > 0:
        expression_padded = F.pad(expression, (0, padding_size), "constant", 0)
    else:
        expression_padded = expression

    output_dict['expression'] = expression_padded

    return output_dict


def concatenate_samples(processed_samples):
    concatenated_dict = {}
    for key in processed_samples[0].keys():
        concatenated_dict[key] = torch.cat([sample[key] for sample in processed_samples], dim=0)
    return concatenated_dict


def process_folder(folder_path, num_expression=100):
    def sort_key(path):
        match = re.search(r'frame_(\d+)', path)
        if match:
            return int(match.group(1))
        else:
            return 0

    seqs = os.listdir(folder_path)
    processed_samples = []
    keep_rate = 0.3
    for seqs in tqdm.tqdm(seqs):
        files = glob.glob(os.path.join(folder_path, seqs, 'body_idx_*', 'results', 'frame_*', '000.pkl'))
        files = sorted(files, key=sort_key)
        files = files[::int(1 / keep_rate)]  # sample partial frames
        for file in files:
            with open(file, 'rb') as f:
                sample = pickle.load(f, encoding='latin1')
            processed_sample = process_sample(sample, num_expression=num_expression)
            processed_samples.append(processed_sample)

    # Now, instead of a list, we create a concatenated dictionary
    concatenated_samples = concatenate_samples(processed_samples)
    print(concatenated_samples['left_hand_pose'].shape)
    # Save the concatenated dictionary as a .pt file
    torch.save(concatenated_samples, '{}.pt'.format(folder_path))

    return concatenated_samples


if __name__ == '__main__':
    print(type(left_hand_model.hand_components))
    print(left_hand_model.hand_components.shape)
    debug()
    # root_dir = '/data3/ljz24/projects/3d/data/human/WholeBodydataset/EgoBody/smplx_parameters'
    # for folder in os.listdir(root_dir):
    #     if os.path.isdir(os.path.join(root_dir, folder)):
    #         process_folder(os.path.join(root_dir, folder))

