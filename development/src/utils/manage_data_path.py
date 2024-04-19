import json
import os

from typing import (
    List,
    Tuple,
)

from tqdm import tqdm


def retrieve_data_path(
    data_root_path: str,
    train_mask_data_path: str,
    val_mask_data_path: str,
    label_file: str,
) -> Tuple[list, list, list, list, list, list]:
    label_file_json_path = os.path.join(data_root_path, label_file)

    with open(label_file_json_path) as file:
        label_data = json.load(file)["root"]

    train_img_list, val_img_list = [], []
    train_mask_list, val_mask_list = [], []
    train_meta_list, val_meta_list = [], []

    for data in label_data:
        img_path = os.path.join(data_root_path, data["img_paths"])
        mask_path = (
            train_mask_data_path + data["img_paths"][-16:-4] + ".jpg"
            if data["isValidation"] == 0
            else val_mask_data_path + data["img_paths"][-16:-4] + ".jpg"
        )

        if data["isValidation"] == 0:
            train_img_list.append(img_path)
            train_mask_list.append(mask_path)
            train_meta_list.append(data)
        else:
            val_img_list.append(img_path)
            val_mask_list.append(mask_path)
            val_meta_list.append(data)

    return (
        train_img_list,
        train_mask_list,
        train_meta_list,
        val_img_list,
        val_mask_list,
        val_meta_list,
    )


def generate_label_file_for_subset_data(meta_list: List[dict], save_file: str) -> str:
    subset_data = {"root": meta_list}
    total_iterations = len(meta_list)

    subset_data_str = json.dumps(subset_data, indent=2)

    with open(save_file, "w") as outfile:
        with tqdm(total=total_iterations, desc="Generating label file") as pbar:
            outfile.write(subset_data_str)
            pbar.update(total_iterations)

    return save_file
