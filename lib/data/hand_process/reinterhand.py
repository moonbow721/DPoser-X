import os
import json
import torch


def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def process_files(folder_path):
    left_files = {}
    right_files = {}

    for file in os.listdir(folder_path):
        if file.endswith("_left.json"):
            left_files[file.split("_left.json")[0]] = file
        elif file.endswith("_right.json"):
            right_files[file.split("_right.json")[0]] = file

    if len(left_files) != len(right_files):
        print("The number of left and right files does not match.")
        return

    merged_data = {
        "left_pose": [], "right_pose": [],
        "left_trans": [], "right_trans": [],
        "left_shape": [], "right_shape": [],
    }

    for key in left_files.keys():
        assert key in right_files, "The left and right files do not match."
        left_data = load_json(os.path.join(folder_path, left_files[key]))
        right_data = load_json(os.path.join(folder_path, right_files[key]))

        for k in merged_data.keys():
            if k.startswith("left"):
                merged_data[k].append(left_data[k.split("left_")[1]])
            elif k.startswith("right"):
                merged_data[k].append(right_data[k.split("right_")[1]])
            else:
                raise ValueError("Invalid key.")

    for k in merged_data.keys():
        merged_data[k] = torch.tensor(merged_data[k])

    torch.save(merged_data,
               "/data3/ljz24/projects/3d/data/human/Handdataset/ReInterHand/all_mano_fits/merged_samples.pt")


if __name__ == '__main__':
    process_files("/data3/ljz24/projects/3d/data/human/Handdataset/ReInterHand/all_mano_fits/params")
