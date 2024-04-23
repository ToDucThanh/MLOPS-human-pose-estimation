import random

from typing import (
    Iterable,
    Tuple,
)

import cv2
import numpy as np
import torch


def get_annotation(
    meta_data: dict, img: np.ndarray, mask_miss: np.ndarray
) -> Tuple[dict, np.ndarray, np.ndarray]:
    anno = dict()
    anno["dataset"] = meta_data["dataset"]
    anno["img_height"] = int(meta_data["img_height"])
    anno["img_width"] = int(meta_data["img_width"])

    anno["isValidation"] = meta_data["isValidation"]
    anno["people_index"] = int(meta_data["people_index"])
    anno["annolist_index"] = int(meta_data["annolist_index"])

    anno["objpos"] = np.array(meta_data["objpos"])
    anno["scale_provided"] = meta_data["scale_provided"]
    anno["joint_self"] = np.array(meta_data["joint_self"])

    anno["numOtherPeople"] = int(meta_data["numOtherPeople"])
    anno["num_keypoints_other"] = np.array(meta_data["num_keypoints_other"])
    anno["joint_others"] = np.array(meta_data["joint_others"])
    anno["objpos_other"] = np.array(meta_data["objpos_other"])
    anno["scale_provided_other"] = meta_data["scale_provided_other"]
    anno["bbox_other"] = meta_data["bbox_other"]
    anno["segment_area_other"] = meta_data["segment_area_other"]

    if anno["numOtherPeople"] == 1:
        anno["joint_others"] = np.expand_dims(anno["joint_others"], 0)
        anno["objpos_other"] = np.expand_dims(anno["objpos_other"], 0)

    return anno, img, mask_miss


def add_neck(
    meta_data: dict, img: np.ndarray, mask_miss: np.ndarray
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    MS COCO annotation order:
    0: nose          1: left eye      2: right eye    3: left ear    4: right ear
    5: left shoulder 6: right shoulder 7: left elbow  8: right elbow
    9: left wrist    10: right wrist  11: left hip    12: right hip   13: left knee
    14: right knee   15: left ankle   16: right ankle
    """

    meta = meta_data
    our_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

    # Index 6 is right shoulder and Index 5 is left shoulder
    right_shoulder = meta["joint_self"][6, :]
    left_shoulder = meta["joint_self"][5, :]
    neck = (right_shoulder + left_shoulder) / 2

    if right_shoulder[2] == 2 or left_shoulder[2] == 2:
        neck[2] = 2
    elif right_shoulder[2] == 1 or left_shoulder[2] == 1:
        neck[2] = 1
    else:
        neck[2] = right_shoulder[2] * left_shoulder[2]

    neck = neck.reshape(1, len(neck))
    neck = np.round(neck)
    meta["joint_self"] = np.vstack((meta["joint_self"], neck))
    meta["joint_self"] = meta["joint_self"][our_order, :]
    temp = []

    for i in range(meta["numOtherPeople"]):
        right_shoulder = meta["joint_others"][i, 6, :]
        left_shoulder = meta["joint_others"][i, 5, :]
        neck = (right_shoulder + left_shoulder) / 2
        if right_shoulder[2] == 2 or left_shoulder[2] == 2:
            neck[2] = 2
        elif right_shoulder[2] == 1 or left_shoulder[2] == 1:
            neck[2] = 1
        else:
            neck[2] = right_shoulder[2] * left_shoulder[2]
        neck = neck.reshape(1, len(neck))
        neck = np.round(neck)
        single_p = np.vstack((meta["joint_others"][i], neck))
        single_p = single_p[our_order, :]
        temp.append(single_p)
    meta["joint_others"] = np.array(temp)

    return meta, img, mask_miss


def aug_scale(
    meta_data: dict, img: np.ndarray, mask_miss: np.ndarray
) -> Tuple[dict, np.ndarray, np.ndarray]:
    params_transform = dict()
    params_transform["scale_min"] = 0.5
    params_transform["scale_max"] = 1.1
    params_transform["target_dist"] = 0.6
    dice = random.random()
    scale_multiplier = (
        params_transform["scale_max"] - params_transform["scale_min"]
    ) * dice + params_transform["scale_min"]

    scale_abs = params_transform["target_dist"] / meta_data["scale_provided"]
    scale = scale_abs * scale_multiplier
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    mask_miss = cv2.resize(mask_miss, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    meta_data["objpos"] *= scale
    meta_data["joint_self"][:, :2] *= scale
    if meta_data["numOtherPeople"] != 0:
        meta_data["objpos_other"] *= scale
        meta_data["joint_others"][:, :, :2] *= scale

    return meta_data, img, mask_miss


def aug_rotate(
    meta_data: dict, img: np.ndarray, mask_miss: np.ndarray
) -> Tuple[dict, np.ndarray, np.ndarray]:
    def rotate_bound(image: np.ndarray, angle: float, bordervalue: float) -> Tuple[np.ndarray]:
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # Grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # Perform the actual rotation and return the image
        return (
            cv2.warpAffine(
                image,
                M,
                (nW, nH),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=bordervalue,
            ),
            M,
        )

    def rotatepoint(p: Iterable[float], R: np.ndarray):
        point = np.zeros((3, 1))
        point[0] = p[0]
        point[1] = p[1]
        point[2] = 1
        new_point = R.dot(point)
        p[0] = new_point[0]
        p[1] = new_point[1]
        return p

    params_transform = dict()
    params_transform["max_rotate_degree"] = 40

    dice = random.random()
    degree = (dice - 0.5) * 2 * params_transform["max_rotate_degree"]  # degree [-40,40]

    img_rot, R = rotate_bound(img, np.copy(degree), (128, 128, 128))
    mask_miss_rot, _ = rotate_bound(mask_miss, np.copy(degree), (255, 255, 255))

    meta_data["objpos"] = rotatepoint(meta_data["objpos"], R)

    for i in range(18):
        meta_data["joint_self"][i, :] = rotatepoint(meta_data["joint_self"][i, :], R)

    for j in range(meta_data["numOtherPeople"]):
        meta_data["objpos_other"][j, :] = rotatepoint(meta_data["objpos_other"][j, :], R)

        for i in range(18):
            meta_data["joint_others"][j, i, :] = rotatepoint(meta_data["joint_others"][j, i, :], R)

    return meta_data, img_rot, mask_miss_rot


def aug_croppad(
    meta_data: dict, img: np.ndarray, mask_miss: np.ndarray
) -> Tuple[dict, np.ndarray, np.ndarray]:
    params_transform = dict()
    params_transform["center_perterb_max"] = 40
    params_transform["crop_size_x"] = 368
    params_transform["crop_size_y"] = 368

    dice_x = random.random()
    dice_y = random.random()
    crop_x = int(params_transform["crop_size_x"])
    crop_y = int(params_transform["crop_size_y"])
    x_offset = int((dice_x - 0.5) * 2 * params_transform["center_perterb_max"])
    y_offset = int((dice_y - 0.5) * 2 * params_transform["center_perterb_max"])

    center = meta_data["objpos"] + np.array([x_offset, y_offset])
    center = center.astype(int)

    # Pad up and down
    pad_v = np.ones((crop_y, img.shape[1], 3), dtype=np.uint8) * 128
    pad_v_mask_miss = np.ones((crop_y, mask_miss.shape[1], 3), dtype=np.uint8) * 255
    img = np.concatenate((pad_v, img, pad_v), axis=0)

    mask_miss = np.concatenate((pad_v_mask_miss, mask_miss, pad_v_mask_miss), axis=0)

    # Pad right and left
    pad_h = np.ones((img.shape[0], crop_x, 3), dtype=np.uint8) * 128
    pad_h_mask_miss = np.ones((mask_miss.shape[0], crop_x, 3), dtype=np.uint8) * 255

    img = np.concatenate((pad_h, img, pad_h), axis=1)
    mask_miss = np.concatenate((pad_h_mask_miss, mask_miss, pad_h_mask_miss), axis=1)

    # Cut
    img = img[
        int(center[1] + crop_y / 2) : int(center[1] + crop_y / 2 + crop_y),
        int(center[0] + crop_x / 2) : int(center[0] + crop_x / 2 + crop_x),
        :,
    ]

    mask_miss = mask_miss[
        int(center[1] + crop_y / 2) : int(center[1] + crop_y / 2 + crop_y + 1 * 0),
        int(center[0] + crop_x / 2) : int(center[0] + crop_x / 2 + crop_x + 1 * 0),
    ]

    offset_left = crop_x / 2 - center[0]
    offset_up = crop_y / 2 - center[1]

    offset = np.array([offset_left, offset_up])
    meta_data["objpos"] += offset
    meta_data["joint_self"][:, :2] += offset

    mask = np.logical_or.reduce(
        (
            meta_data["joint_self"][:, 0] >= crop_x,
            meta_data["joint_self"][:, 0] < 0,
            meta_data["joint_self"][:, 1] >= crop_y,
            meta_data["joint_self"][:, 1] < 0,
        )
    )

    meta_data["joint_self"][mask, 2] = 2
    if meta_data["numOtherPeople"] != 0:
        meta_data["objpos_other"] += offset
        meta_data["joint_others"][:, :, :2] += offset

        mask = np.logical_or.reduce(
            (
                meta_data["joint_others"][:, :, 0] >= crop_x,
                meta_data["joint_others"][:, :, 0] < 0,
                meta_data["joint_others"][:, :, 1] >= crop_y,
                meta_data["joint_others"][:, :, 1] < 0,
            )
        )

        meta_data["joint_others"][mask, 2] = 2

    return meta_data, img, mask_miss


def croppad(
    meta_data: dict, img: np.ndarray, mask_miss: np.ndarray
) -> Tuple[dict, np.ndarray, np.ndarray]:
    params_transform = dict()
    params_transform["center_perterb_max"] = 40
    params_transform["crop_size_x"] = 368
    params_transform["crop_size_y"] = 368

    crop_x = int(params_transform["crop_size_x"])
    crop_y = int(params_transform["crop_size_y"])

    center = meta_data["objpos"]
    center = center.astype(int)

    # Pad up and down
    pad_v = np.ones((crop_y, img.shape[1], 3), dtype=np.uint8) * 128
    pad_v_mask_miss = np.ones((crop_y, mask_miss.shape[1], 3), dtype=np.uint8) * 255
    img = np.concatenate((pad_v, img, pad_v), axis=0)

    mask_miss = np.concatenate((pad_v_mask_miss, mask_miss, pad_v_mask_miss), axis=0)

    # Pad right and left
    pad_h = np.ones((img.shape[0], crop_x, 3), dtype=np.uint8) * 128
    pad_h_mask_miss = np.ones((mask_miss.shape[0], crop_x, 3), dtype=np.uint8) * 255

    img = np.concatenate((pad_h, img, pad_h), axis=1)
    mask_miss = np.concatenate((pad_h_mask_miss, mask_miss, pad_h_mask_miss), axis=1)

    # Cut
    img = img[
        int(center[1] + crop_y / 2) : int(center[1] + crop_y / 2 + crop_y),
        int(center[0] + crop_x / 2) : int(center[0] + crop_x / 2 + crop_x),
        :,
    ]

    mask_miss = mask_miss[
        int(center[1] + crop_y / 2) : int(center[1] + crop_y / 2 + crop_y + 1 * 0),
        int(center[0] + crop_x / 2) : int(center[0] + crop_x / 2 + crop_x + 1 * 0),
    ]

    offset_left = crop_x / 2 - center[0]
    offset_up = crop_y / 2 - center[1]

    offset = np.array([offset_left, offset_up])
    meta_data["objpos"] += offset
    meta_data["joint_self"][:, :2] += offset

    mask = np.logical_or.reduce(
        (
            meta_data["joint_self"][:, 0] >= crop_x,
            meta_data["joint_self"][:, 0] < 0,
            meta_data["joint_self"][:, 1] >= crop_y,
            meta_data["joint_self"][:, 1] < 0,
        )
    )

    meta_data["joint_self"][mask, 2] = 2
    if meta_data["numOtherPeople"] != 0:
        meta_data["objpos_other"] += offset
        meta_data["joint_others"][:, :, :2] += offset

        mask = np.logical_or.reduce(
            (
                meta_data["joint_others"][:, :, 0] >= crop_x,
                meta_data["joint_others"][:, :, 0] < 0,
                meta_data["joint_others"][:, :, 1] >= crop_y,
                meta_data["joint_others"][:, :, 1] < 0,
            )
        )

        meta_data["joint_others"][mask, 2] = 2

    return meta_data, img, mask_miss


def aug_flip(
    meta_data: dict, img: np.ndarray, mask_miss: np.ndarray
) -> Tuple[dict, np.ndarray, np.ndarray]:
    params_transform = dict()
    params_transform["flip_prob"] = 0.5

    dice = random.random()
    doflip = dice <= params_transform["flip_prob"]

    if doflip:
        img = img.copy()
        cv2.flip(src=img, flipCode=1, dst=img)
        w = img.shape[1]

        mask_miss = mask_miss.copy()
        cv2.flip(src=mask_miss, flipCode=1, dst=mask_miss)

        meta_data["objpos"][0] = w - 1 - meta_data["objpos"][0]
        meta_data["joint_self"][:, 0] = w - 1 - meta_data["joint_self"][:, 0]
        meta_data["joint_self"] = meta_data["joint_self"][
            [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
        ]

        num_other_people = meta_data["numOtherPeople"]
        if num_other_people != 0:
            meta_data["objpos_other"][:, 0] = w - 1 - meta_data["objpos_other"][:, 0]
            meta_data["joint_others"][:, :, 0] = w - 1 - meta_data["joint_others"][:, :, 0]
            for i in range(num_other_people):
                meta_data["joint_others"][i] = meta_data["joint_others"][i][
                    [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
                ]

    return meta_data, img, mask_miss


def remove_illegal_joint(
    meta_data: dict, img: np.ndarray, mask_miss: np.ndarray
) -> Tuple[dict, np.ndarray, np.ndarray]:
    params_transform = dict()
    params_transform["crop_size_x"] = 368
    params_transform["crop_size_y"] = 368

    crop_x = int(params_transform["crop_size_x"])
    crop_y = int(params_transform["crop_size_y"])

    mask = np.logical_or.reduce(
        (
            meta_data["joint_self"][:, 0] >= crop_x,
            meta_data["joint_self"][:, 0] < 0,
            meta_data["joint_self"][:, 1] >= crop_y,
            meta_data["joint_self"][:, 1] < 0,
        )
    )

    # Get parts
    meta_data["joint_self"][mask, :] = (1, 1, 2)

    if meta_data["numOtherPeople"] != 0:
        mask = np.logical_or.reduce(
            (
                meta_data["joint_others"][:, :, 0] >= crop_x,
                meta_data["joint_others"][:, :, 0] < 0,
                meta_data["joint_others"][:, :, 1] >= crop_y,
                meta_data["joint_others"][:, :, 1] < 0,
            )
        )
        meta_data["joint_others"][mask, :] = (1, 1, 2)

    return meta_data, img, mask_miss


class TensorNormalization:
    def __init__(self):
        self.color_mean = [0.485, 0.456, 0.406]
        self.color_std = [0.229, 0.224, 0.225]

    def __call__(
        self, meta_data: dict, img: np.ndarray, mask_miss: np.ndarray
    ) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        img = img.astype(np.float32) / 255.0

        preprocessed_img = img.copy()[:, :, ::-1]  # BGR→RGB

        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - self.color_mean[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / self.color_std[i]

        img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
        mask_miss = mask_miss.transpose((2, 0, 1)).astype(np.float32)

        img = torch.from_numpy(img)
        mask_miss = torch.from_numpy(mask_miss)

        return meta_data, img, mask_miss


class TensorNotNormalization(TensorNormalization):
    def __init__(self):
        super().__init__()
        self.color_mean = [0, 0, 0]
        self.color_std = [1, 1, 1]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(
        self, meta_data: dict, img: np.ndarray, mask_miss: np.ndarray
    ) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            meta_data, img, mask_miss = transform(meta_data, img, mask_miss)
        return meta_data, img, mask_miss


class DataTransform:
    def __init__(self):
        self.data_transform = {
            "train": Compose(
                [
                    get_annotation,
                    add_neck,
                    aug_scale,
                    aug_rotate,
                    aug_croppad,
                    aug_flip,
                    remove_illegal_joint,
                    TensorNormalization(),
                ]
            ),
            "val": Compose(
                [
                    get_annotation,
                    add_neck,
                    croppad,
                    remove_illegal_joint,
                    TensorNormalization(),
                ]
            ),
        }

    def __call__(
        self, phase: str, meta_data: dict, img: np.ndarray, mask_miss: np.ndarray
    ) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        meta_data, img, mask_miss = self.data_transform[phase](meta_data, img, mask_miss)
        return meta_data, img, mask_miss
