import numpy as np
import cv2
import torch
from plyfile import PlyData

from lib.body_model import constants
from lib.body_model.joint_mapping import get_openpose_part


# codes from https://github.com/mks0601/Hand4Whole_RELEASE/blob/main/common/utils/preprocessing.py
# and https://github.com/haofanwang/CLIFF/blob/main/common/imutils.py
def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img


def load_obj(file_name):
    v = []
    obj_file = open(file_name)
    for line in obj_file:
        words = line.split(' ')
        if words[0] == 'v':
            x, y, z = float(words[1]), float(words[2]), float(words[3])
            v.append(np.array([x, y, z]))
    return np.stack(v)


def load_ply(file_name):
    plydata = PlyData.read(file_name)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    v = np.stack((x, y, z), 1)
    return v


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    # res: (height, width), (rows, cols)
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1


def crop(img, center, scale, res):
    """
    Crop image according to the supplied bounding box.
    res: [rows, cols]
    """
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    except Exception as e:
        print(e)

    new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

    return new_img, ul, br


def bbox_from_detector(bbox, rescale=1.1):
    """
    Get center and scale of bounding box from bounding box.
    The expected format is [min_x, min_y, max_x, max_y].
    """
    # center
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    center = torch.tensor([center_x, center_y])

    # scale
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_size = max(bbox_w * constants.CROP_ASPECT_RATIO, bbox_h)
    scale = bbox_size / 200.0
    # adjust bounding box tightness
    scale *= rescale
    return center, scale


def compute_bbox(keypoints_list):
    all_keypoints = keypoints_list
    bbox_list = []

    for batch_id, keypoints in enumerate(all_keypoints):
        visible_keypoints = keypoints[keypoints[:, 2] > 0]

        if len(visible_keypoints) == 0:
            continue

        x_coords = visible_keypoints[:, 0]
        y_coords = visible_keypoints[:, 1]

        min_x = np.min(x_coords)
        min_y = np.min(y_coords)
        max_x = np.max(x_coords)
        max_y = np.max(y_coords)

        # [batch_id, min_x, min_y, max_x, max_y]
        bbox = [batch_id, min_x, min_y, max_x, max_y]
        bbox_list.append(bbox)

    bbox_array = np.array(bbox_list)
    return bbox_array


def get_best_hand(keypoints_list, hand='rhand', from_wholebody=True):
    hand_idx = get_openpose_part(hand)
    all_conf = []
    for keypoints in keypoints_list:
        hand_keypoints = keypoints[hand_idx] if from_wholebody else keypoints
        mean_conf = np.mean(hand_keypoints[:, 2])
        # sanity check
        x_coords = hand_keypoints[:, 0]
        y_coords = hand_keypoints[:, 1]
        if abs(np.max(x_coords) - np.min(x_coords)) < 2.0 or abs(np.max(y_coords) - np.min(y_coords)) < 2.0:
            mean_conf = 0
        all_conf.append(mean_conf)
    if np.max(all_conf) == 0:
        raise ValueError('All hand keypoints are invisible')
    best_idx = np.argmax(all_conf)

    return best_idx


def get_best_face(keypoints_list, from_wholebody=True):
    face_idx = get_openpose_part('face')
    all_area = []
    for keypoints in keypoints_list:
        face_keypoints = keypoints[face_idx] if from_wholebody else keypoints
        face_area = (face_keypoints[:, 0].max() - face_keypoints[:, 0].min()) * (
                face_keypoints[:, 1].max() - face_keypoints[:, 1].min())
        all_area.append(face_area)
    best_idx = np.argmax(all_area)

    return best_idx

def process_image(orig_img_rgb, bbox,
                  crop_height=constants.CROP_IMG_HEIGHT,
                  crop_width=constants.CROP_IMG_WIDTH):
    """
    Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    try:
        center, scale = bbox_from_detector(bbox)
    except Exception as e:
        print("Error occurs in person detection", e)
        # Assume that the person is centered in the image
        height = orig_img_rgb.shape[0]
        width = orig_img_rgb.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width * crop_height / float(crop_width)) / 200.

    img, ul, br = crop(orig_img_rgb, center, scale, (crop_height, crop_width))
    crop_img = img.copy()

    img = img / 255.
    mean = np.array(constants.IMG_NORM_MEAN, dtype=np.float32)
    std = np.array(constants.IMG_NORM_STD, dtype=np.float32)
    norm_img = (img - mean) / std
    norm_img = np.transpose(norm_img, (2, 0, 1))

    return norm_img, center, scale, ul, br, crop_img


def crop_img(ori_image, rect, cropped_size, kpts=None):
    image = ori_image.copy()
    l, t, r, b = rect
    center_x = int(r - (r - l) // 2)
    center_y = int(b - (b - t) // 2)
    w = int((r - l) * 1.2)
    h = int((b - t) * 1.2)
    crop_size = max(w, h, cropped_size)  # Ensure crop_size is at least as large as cropped_size

    # Calculate padding needs
    pad_top = max(0, -(center_y - crop_size // 2))
    pad_left = max(0, -(center_x - crop_size // 2))
    pad_bottom = max(0, (center_y + crop_size // 2) - image.shape[0])
    pad_right = max(0, (center_x + crop_size // 2) - image.shape[1])

    # Pad the image if necessary
    if any([pad_top, pad_left, pad_bottom, pad_right]):
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    # Update center coordinates to new padded image dimensions
    center_y += pad_top
    center_x += pad_left

    # Recalculate crop coordinates based on padded image
    crop_ly = int(center_y - crop_size // 2)
    crop_lx = int(center_x - crop_size // 2)

    crop_image = image[crop_ly: crop_ly + crop_size, crop_lx: crop_lx + crop_size, :]

    # Calculate the new rectangle in the cropped image coordinates
    new_rect = [l + pad_left - crop_lx, t + pad_top - crop_ly, r + pad_left - crop_lx, b + pad_top - crop_ly]

    # Adjust keypoints if provided
    if kpts is not None:
        new_kpts = kpts.copy()
        new_kpts[:, 0] = kpts[:, 0] + pad_left - crop_lx
        new_kpts[:, 1] = kpts[:, 1] + pad_top - crop_ly
        return crop_image, new_rect, new_kpts

    return crop_image, new_rect


def crop_img_tensor(ori_image_tensor, rect, cropped_size, kpts=None):
    image_tensor = ori_image_tensor.clone()
    l, t, r, b = rect
    center_x = int(r - (r - l) // 2)
    center_y = int(b - (b - t) // 2)
    w = int((r - l) * 1.2)
    h = int((b - t) * 1.2)
    crop_size = max(w, h, cropped_size)  # Ensure crop_size is at least as large as cropped_size

    # Calculate padding needs
    pad_top = max(0, -(center_y - crop_size // 2))
    pad_left = max(0, -(center_x - crop_size // 2))
    pad_bottom = max(0, (center_y + crop_size // 2) - image_tensor.shape[1])
    pad_right = max(0, (center_x + crop_size // 2) - image_tensor.shape[2])

    # Pad the image if necessary using 'constant' mode equivalent in PyTorch
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    image_tensor = torch.nn.functional.pad(image_tensor, padding, mode='constant', value=0)

    # Update center coordinates to new padded image dimensions
    center_y += pad_top
    center_x += pad_left

    # Recalculate crop coordinates based on padded image
    crop_ly = int(center_y - crop_size // 2)
    crop_lx = int(center_x - crop_size // 2)

    # Crop the image using PyTorch tensor indexing
    crop_image_tensor = image_tensor[:, crop_ly: crop_ly + crop_size, crop_lx: crop_lx + crop_size]

    # Calculate the new rectangle in the cropped image coordinates
    new_rect = [l + pad_left - crop_lx, t + pad_top - crop_ly, r + pad_left - crop_lx, b + pad_top - crop_ly]

    # Adjust keypoints if provided
    if kpts is not None:
        new_kpts = kpts.copy()
        new_kpts[:, 0] = kpts[:, 0] + pad_left - crop_lx
        new_kpts[:, 1] = kpts[:, 1] + pad_top - crop_ly
        return crop_image_tensor, new_rect, new_kpts

    return crop_image_tensor, new_rect