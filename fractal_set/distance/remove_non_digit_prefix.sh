#!/bin/bash
# 将当前文件夹下所有jpg文件移动到raw子文件夹
folder="/mnt/d/code/nano/optmized_fractal/fractal_set/occ5"
raw_folder="$folder/raw"
mkdir -p "$raw_folder"
find "$folder" -maxdepth 1 -type f -name "*.jpg" -exec mv -v {} "$raw_folder" \;
