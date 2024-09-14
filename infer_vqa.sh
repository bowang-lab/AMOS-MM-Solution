#!/bin/bash --login

#SBATCH --job-name=vqa_infer
#SBATCH --output=runs/infer_vqa-log-%J.txt
#SBATCH --ntasks=6
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#/SBATCH --partition=a100
#/SBATCH --qos=a100_bowang
#SBATCH --gres=gpu:a40:1

source /h/junma/.mvlm/bin/activate
CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29500 infer_vqa.py \
  --model_name_or_path  /scratch/ssd004/datasets/med-img-data/amosmm/trained/download/llama_vqa_gen_all_template \
  --json_path /fs01/home/junma/MedicalVLM/Data/AMOSMM.json \
  --image_size 32 256 256 \
  --model_max_length 512 \
  --proj_out_num 256
