#!/bin/sh
export LD_LIBRARY_PATH=/home/seu/miniconda3/envs/NTIRE23/lib/python3.9/site-packages/torch/lib/../../nvidia/cublas/lib/:$LD_LIBRARY_PATH
eval "$(conda shell.bash hook)"
conda activate NTIRE23
ROOT=$(dirname $0)

rm -rf "$ROOT/results"
mkdir -p "$ROOT/results"
CUDA_VISIBLE_DEVICES=7 python test_prune.py \
  --lr_dir /home/data/dataset/DIV2K/DIV2K_valid_LR_bicubic/X4 \
  --hr_dir /home/data/dataset/DIV2K/DIV2K_valid_HR \
  --save_dir "$ROOT/results" \
  --model_id -11 \
  --upscale 4
