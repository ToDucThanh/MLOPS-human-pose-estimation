import argparse
import os

from config import cfg
from logs import log
from src.utils.manage_data_path import (
    generate_label_file_for_subset_data,
    retrieve_data_path,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating label file for subset data")
    parser.add_argument(
        "--num_samples",
        metavar="N",
        type=int,
        help="Number of samples to generate subset data",
    )
    args = parser.parse_args()
    num_sample = args.num_samples
    (
        train_img_list,
        train_mask_list,
        train_meta_list,
        val_img_list,
        val_mask_list,
        val_meta_list,
    ) = retrieve_data_path(
        cfg.data_root_path,
        cfg.train_mask_data_path,
        cfg.val_mask_data_path,
        cfg.label_file,
    )

    sample_meta_list = val_meta_list[:num_sample]
    save_output_file = os.path.join(cfg.data_root_path, cfg.label_subset_file)
    output_label_file_path = generate_label_file_for_subset_data(
        sample_meta_list, save_output_file
    )
    log.info(f"The output label json file path: {output_label_file_path}")
