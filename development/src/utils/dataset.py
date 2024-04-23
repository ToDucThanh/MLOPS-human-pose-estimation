from typing import (
    List,
    Tuple,
)

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


def put_gaussian_maps(
    center: List[float],
    accumulate_confid_map: np.ndarray,
    params_transform: dict,
) -> np.ndarray:
    crop_size_y = params_transform["crop_size_y"]
    crop_size_x = params_transform["crop_size_x"]
    stride = params_transform["stride"]
    sigma = params_transform["sigma"]

    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    start = stride / 2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_confid_map += cofid_map
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0

    return accumulate_confid_map


def put_vec_maps(
    centerA: List[float],
    centerB: List[float],
    accumulate_vec_map: np.ndarray,
    count: np.ndarray,
    params_transform: dict,
) -> Tuple[np.ndarray, int]:
    stride = params_transform["stride"]
    crop_size_y = params_transform["crop_size_y"]
    crop_size_x = params_transform["crop_size_x"]
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    thre = params_transform["limb_width"]
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if norm == 0.0:
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)

    # the vector from (x, y) to centerA
    ba_x = xx - centerA[0]
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

    mask = np.logical_or.reduce((np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

    accumulate_vec_map = np.multiply(accumulate_vec_map, count[:, :, np.newaxis])
    accumulate_vec_map += vec_map
    count[mask] += 1

    mask = count == 0

    count[mask] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis])
    count[mask] = 0

    return accumulate_vec_map, count


def get_ground_truth(
    meta: dict, mask_miss: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params_transform = dict()
    params_transform["stride"] = 8
    params_transform["mode"] = 5
    params_transform["crop_size_x"] = 368
    params_transform["crop_size_y"] = 368
    params_transform["np"] = 56
    params_transform["sigma"] = 7.0
    params_transform["limb_width"] = 1.0

    stride = params_transform["stride"]
    crop_size_y = params_transform["crop_size_y"]
    crop_size_x = params_transform["crop_size_x"]
    nop = meta["numOtherPeople"]

    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride

    heatmaps = np.zeros((int(grid_y), int(grid_x), 19))
    pafs = np.zeros((int(grid_y), int(grid_x), 38))

    mask_miss = cv2.resize(
        mask_miss, (0, 0), fx=1.0 / stride, fy=1.0 / stride, interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)
    mask_miss = mask_miss / 255.0
    mask_miss = np.expand_dims(mask_miss, axis=2)

    heat_mask = np.repeat(mask_miss, 19, axis=2)
    paf_mask = np.repeat(mask_miss, 38, axis=2)

    for i in range(18):
        if meta["joint_self"][i, 2] <= 1:
            center = meta["joint_self"][i, :2]
            gaussian_map = heatmaps[:, :, i]
            heatmaps[:, :, i] = put_gaussian_maps(center, gaussian_map, params_transform)
        for j in range(nop):
            if meta["joint_others"][j, i, 2] <= 1:
                center = meta["joint_others"][j, i, :2]
                gaussian_map = heatmaps[:, :, i]
                heatmaps[:, :, i] = put_gaussian_maps(center, gaussian_map, params_transform)
    # PAFs
    mid_1 = [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16]
    mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]

    for i in range(19):
        count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
        if meta["joint_self"][mid_1[i] - 1, 2] <= 1 and meta["joint_self"][mid_2[i] - 1, 2] <= 1:
            centerA = meta["joint_self"][mid_1[i] - 1, :2]
            centerB = meta["joint_self"][mid_2[i] - 1, :2]
            vec_map = pafs[:, :, 2 * i : 2 * i + 2]

            pafs[:, :, 2 * i : 2 * i + 2], count = put_vec_maps(
                centerA=centerA,
                centerB=centerB,
                accumulate_vec_map=vec_map,
                count=count,
                params_transform=params_transform,
            )
        for j in range(nop):
            if (
                meta["joint_others"][j, mid_1[i] - 1, 2] <= 1
                and meta["joint_others"][j, mid_2[i] - 1, 2] <= 1
            ):
                centerA = meta["joint_others"][j, mid_1[i] - 1, :2]
                centerB = meta["joint_others"][j, mid_2[i] - 1, :2]
                vec_map = pafs[:, :, 2 * i : 2 * i + 2]
                pafs[:, :, 2 * i : 2 * i + 2], count = put_vec_maps(
                    centerA=centerA,
                    centerB=centerB,
                    accumulate_vec_map=vec_map,
                    count=count,
                    params_transform=params_transform,
                )
    # background
    heatmaps[:, :, -1] = np.maximum(1 - np.max(heatmaps[:, :, :18], axis=2), 0.0)

    heat_mask = torch.from_numpy(heat_mask)
    heatmaps = torch.from_numpy(heatmaps)
    paf_mask = torch.from_numpy(paf_mask)
    pafs = torch.from_numpy(pafs)

    return heat_mask, heatmaps, paf_mask, pafs


class COCOKeypointsDataset(Dataset):
    def __init__(
        self,
        img_list: List,
        mask_list: List,
        meta_list: List,
        phase: str,
        transform,
    ):
        self.img_list = img_list
        self.mask_list = mask_list
        self.meta_list = meta_list
        self.phase = phase
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        img, heatmaps, heat_mask, pafs, paf_mask = self.get_item(index)
        return img, heatmaps, heat_mask, pafs, paf_mask

    def get_item(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)

        mask_miss = cv2.imread(self.mask_list[index])
        meat_data = self.meta_list[index]

        meta_data, img, mask_miss = self.transform(self.phase, meat_data, img, mask_miss)

        mask_miss_numpy = mask_miss.numpy().transpose((1, 2, 0))
        heat_mask, heatmaps, paf_mask, pafs = get_ground_truth(meta_data, mask_miss_numpy)

        heat_mask = heat_mask[:, :, :, 0]
        paf_mask = paf_mask[:, :, :, 0]

        paf_mask = paf_mask.permute(2, 0, 1)
        heat_mask = heat_mask.permute(2, 0, 1)
        pafs = pafs.permute(2, 0, 1)
        heatmaps = heatmaps.permute(2, 0, 1)

        return img, heatmaps, heat_mask, pafs, paf_mask
