#!/bin/bash

#SBATCH --job-name=JunAbd
#SBATCH --output=runs/JunAbd-log-%J.txt
#SBATCH --ntasks=16
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --qos=a100_bowang

# run "accelerate config" first!
JOB_ID=$SLURM_JOB_ID

#  torchrun --nproc_per_node=1
# CUDA_VISIBLE_DEVICES="0" PYTHONPATH=. accelerate launch --main_process_port 0
PYTHONPATH=. CUDA_VISIBLE_DEVICES="0" accelerate launch --main_process_port 29501 LaMed/src/train/abdomen_train.py \
    --version v0 \
    --model_name_or_path /scratch/ssd004/datasets/med-img-data/amosmm/LaMed/M3D-LaMed-Phi-3-4B \
    --model_type phi3 \
    --lora_enable True \
    --vision_tower vit3d \
    --pretrain_vision_model /scratch/ssd004/datasets/med-img-data/amosmm/LaMed/M3D-CLIP/pretrained_ViT.bin \
    --bf16 True \
    --output_dir /scratch/ssd004/datasets/med-img-data/amosmm/trained/junAbdomen/ \
    --num_train_epochs 200 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --do_eval False \
    --eval_accumulation_steps 1 \
    --eval_steps 0.99 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to none \
    --use_hilt False \
    --task mrg \
    --image_size "32, 256, 256" \
    --patch_size "4, 16, 16" \
    --model_max_length 512

