#!/bin/bash

data_dir="./data/raw"

val_data_link="http://images.cocodataset.org/zips/val2014.zip"
train_data_link="http://images.cocodataset.org/zips/train2014.zip"

target_val_data_path="$data_dir/val2014.zip"
target_train_data_path="$data_dir/train2014.zip"

label_link="https://www.dropbox.com/s/0sj2q24hipiiq5t/COCO.json?dl=1"
target_label_path="$data_dir/label.json"

mask_data_link="https://www.dropbox.com/s/bd9ty7b4fqd5ebf/mask.tar.gz?dl=0"
mask_data_path="$data_dir/mask.tar.gz"


if [ ! -d "$data_dir" ]; then
    mkdir -p "$data_dir"
fi


## Download label file
if [ ! -f "$target_label_path" ]; then
    wget -O "$target_label_path" "$label_link"
fi

## Download mask data
if [ ! -f "$mask_data_path" ]; then
    wget -O "$mask_data_path" "$mask_data_link"
    tar -xzvf "$mask_data_path" -C "$data_dir"
fi

## Download and unzip val data
if [ ! -f "$target_val_data_path" ]; then
    wget -O "$target_val_data_path" "$val_data_link"
    unzip "$target_val_data_path" -d "$data_dir"
fi

## Download and unzip train data
if [ ! -f "$target_train_data_path" ]; then
    wget -O "$target_train_data_path" "$train_data_link"
    unzip "$target_train_data_path" -d "$data_dir"
fi
