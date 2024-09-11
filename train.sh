#!/bin/bash

#SBATCH --job-name=run
#SBATCH --output=runs/run-log-%J.txt
#SBATCH --ntasks=32
#SBATCH --time=12:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:a40:4


source /h/mohammed/m3d/bin/activate

# run "accelerate config" first!
JOB_ID=$SLURM_JOB_ID

PYTHONPATH=. accelerate launch LaMed/src/train/amos_train.py \
    --version v0 \
    --model_name_or_path /scratch/ssd004/scratch/mohammed/models/M3D-LaMed-Phi-3-4B \
    --model_type phi3 \
    --lora_enable True \
    --vision_tower vit3d \
    --pretrain_vision_model /scratch/ssd004/scratch/mohammed/models/M3D-CLIP/pretrained_ViT.bin \
    --bf16 True \
    --output_dir /scratch/ssd004/scratch/mohammed/results/hilt_64_320_1024 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --do_eval False \
    --eval_accumulation_steps 1 \
    --eval_steps 0.99 \
    --save_strategy "steps" \
    --save_steps 6000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --use_hilt True \
    --image_size "64, 320, 320" \
    --lr_image_size "32, 128, 128" \
    --model_max_length 768 
