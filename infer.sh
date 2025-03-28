#!/bin/bash --login

#SBATCH --job-name=infer
#SBATCH --output=runs/infer-log-%J.txt
#SBATCH --ntasks=8
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --mem=30GB
#/SBATCH --gres=gpu:1
#/sbatch --n=1
#/SBATCH --partition=a100
#/SBATCH --qos=a100_bowang
#SBATCH --gres=gpu:a40:1

source /h/junma/.mllm/bin/activate
# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29565 infer.py \
#   --model_name_or_path /scratch/ssd004/datasets/med-img-data/amosmm/trained/paper/phi3_150_default  \
#   --json_path Data/AMOSMM.json \
#   --model_max_length 768 \
#   --prompt "simple" \
#   --proj_out_num 256

# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29575 infer.py \
#   --model_name_or_path /scratch/ssd004/datasets/med-img-data/amosmm/trained/paper/phi3_150_finetune  \
#   --json_path Data/AMOSMM.json \
#   --model_max_length 768 \
#   --prompt "simple" \
#   --proj_out_num 256

# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29585 infer.py \
#   --model_name_or_path /scratch/ssd004/datasets/med-img-data/amosmm/trained/paper/llama31_default  \
#   --json_path Data/AMOSMM.json \
#   --model_max_length 768 \
#   --prompt "simple" \
#   --proj_out_num 256


# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29585 infer.py \
#   --model_name_or_path /scratch/ssd004/datasets/med-img-data/amosmm/trained/paper/gemma_default  \
#   --json_path Data/AMOSMM.json \
#   --model_max_length 768 \
#   --prompt "simple" \
#   --proj_out_num 256

# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29585 infer.py \
#   --model_name_or_path /scratch/ssd004/datasets/med-img-data/amosmm/trained/paper/phi3_150_tp  \
#   --json_path Data/AMOSMM.json \
#   --model_max_length 768 \
#   --prompt "simple" \
#   --proj_out_num 256

# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29585 infer.py \
#   --model_name_or_path /scratch/ssd004/datasets/med-img-data/amosmm/trained/paper/phi3_150_lora  \
#   --json_path Data/AMOSMM.json \
#   --model_max_length 768 \
#   --prompt "simple" \
#   --proj_out_num 256

# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29585 infer.py \
#   --model_name_or_path /scratch/ssd004/datasets/med-img-data/amosmm/trained/paper/phi3_150_mlp  \
#   --json_path Data/AMOSMM.json \
#   --model_max_length 768 \
#   --prompt "simple" \
#   --proj_out_num 256

# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29585 infer.py \
#   --model_name_or_path /scratch/ssd004/datasets/med-img-data/amosmm/trained/paper/phi3_150_highres  \
#   --json_path Data/AMOSMM.json \
#   --model_max_length 768 \
#   --prompt "simple" \
#   --proj_out_num 256

# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29585 infer.py \
#   --model_name_or_path /scratch/ssd004/datasets/med-img-data/amosmm/trained/paper/gemma1_default  \
#   --json_path Data/AMOSMM.json \
#   --model_max_length 768 \
#   --prompt "simple" \
#   --proj_out_num 256

# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29505 infer.py \
#   --model_name_or_path /checkpoint/datasets.damaged/med-img-data/amosmm/trained/paper/qwen25_150_3b  \
#   --json_path Data/AMOSMM_corr.json \
#   --model_max_length 768 \
#   --prompt "simple" \
#   --proj_out_num 256

# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29506 infer.py \
#   --model_name_or_path /checkpoint/datasets.damaged/med-img-data/amosmm/trained/paper/phi3_150_with_impressions  \
#   --json_path Data/AMOSMM_corr.json \
#   --model_max_length 768 \
#   --prompt "simple" \
#   --proj_out_num 256

CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29500 infer.py \
  --model_name_or_path /home/jma/Documents/mohammed/amosmm/models/mistral_150_7bv3  \
  --json_path Data/AMOSMM.json \
  --model_max_length 768 \
  --prompt "simple" \
  --proj_out_num 256

CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29510 infer.py \
  --model_name_or_path /home/jma/Documents/mohammed/amosmm/models/phi3_150_with_seg2 \
  --json_path Data/AMOSMM.json \
  --model_max_length 1024 \
  --prompt "simple" \
  --proj_out_num 256

# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29560 infer.py \
#   --model_name_or_path /checkpoint/datasets.damaged/med-img-data/amosmm/trained/paper/phi3_150_default  \
#   --json_path Data/AMOSMM.json \
#   --model_max_length 768 \
#   --post_process "normality" \
#   --prompt "simple" \
#   --proj_out_num 256

# CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29565 infer.py \
#   --model_name_or_path /checkpoint/datasets.damaged/med-img-data/amosmm/trained/paper/phi3_150_default  \
#   --json_path Data/AMOSMM.json \
#   --model_max_length 768 \
#   --prompt "simple" \
#   --organs "chest" \
#   --proj_out_num 256