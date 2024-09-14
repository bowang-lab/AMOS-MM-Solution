#!/bin/bash --login

#SBATCH --job-name=infer
#SBATCH --output=runs/infer-log-%J.txt
#SBATCH --ntasks=8
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#/sbatch --n=1
#SBATCH --partition=a100
#SBATCH --qos=a100_bowang
#/SBATCH --gres=gpu:a40:1

source /h/junma/.mvlm/bin/activate
CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29500 infer.py \
  --model_name_or_path /scratch/ssd004/datasets/med-img-data/amosmm/trained/download/llama3_frozen_simple_all_template \
  --json_path Data/AMOSMM.json \
  --model_max_length 768 \
  --prompt "simple" \
  --post_process "normality" "focused_inference" \
  --proj_out_num 256
