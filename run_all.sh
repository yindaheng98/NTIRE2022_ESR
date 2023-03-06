#!/bin/sh
export LD_LIBRARY_PATH=/home/seu/miniconda3/envs/NTIRE23/lib/python3.9/site-packages/torch/lib/../../nvidia/cublas/lib/:$LD_LIBRARY_PATH
eval "$(conda shell.bash hook)"
conda activate NTIRE23
ROOT=$(dirname $0)

for i in {0..44}; do
rm -rf "$ROOT/results"
mkdir -p "$ROOT/results"
CUDA_VISIBLE_DEVICES=1 python test_demo.py \
  --lr_dir /home/data/dataset/DIV2K/DIV2K_valid_LR_bicubic/X4 \
  --hr_dir /home/data/dataset/DIV2K/DIV2K_valid_HR \
  --save_dir "$ROOT/results" \
  --model_id -2
done
printf "%20s %12s %17s %14s %5s\n" model_name valid_memory valid_ave_runtime valid_ave_psnr flops
for line in $(cat results.json | jq -r 'to_entries|.[]|[.key,.value.valid_memory,.value.valid_ave_runtime,.value.valid_ave_psnr,.value.flops|tostring] | join(",")'); do
  printf "%20s %12f %17f %14f %5f\n" $(echo $line | sed 's/,/ /g')
done
