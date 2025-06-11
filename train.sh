#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=runs/run-log-%J.txt
#SBATCH --ntasks=10
#SBATCH -N 1
#SBATCH --time=1-12:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --qos=a100_bowang

source /home/jma/datasets/mohammed/.mllm/bin/activate

# run "accelerate config" first!
JOB_ID=$SLURM_JOB_ID
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. accelerate launch --num_processes 1 --main_process_port 29500 LaMed/src/train/amos_train.py \
    --version v0 \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --cache_dir /home/jma/datasets/mohammed/FLARE-Task5-MLLM-3D/models \
    --model_type phi3 \
    --lora_enable True \
    --lora_r 16 \
    --vision_tower vit3d \
    --pretrain_vision_model /home/jma/datasets/mohammed/amosmm/models/pretrained_ViT.bin \
    --bf16 True \
    --output_dir  /home/jma/datasets/mohammed/FLARE-Task5-MLLM-3D/results/baseline \
    --num_train_epochs 50 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --do_eval False \
    --eval_accumulation_steps 1 \
    --eval_steps 0.99 \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    --report_to none \
    --prompt "simple" \
    --task mrg \
    --json_path /home/jma/datasets/mohammed/FLARE-Task5-MLLM-3D/train/CT-AMOS-Tr_processed.json /home/jma/datasets/mohammed/FLARE-Task5-MLLM-3D/train/CT-RATE-Tr_processed.json \
    --data_root /home/jma/datasets/mohammed/FLARE-Task5-MLLM-3D/train/CT-AMOS-1290_processed /home/jma/datasets/mohammed/FLARE-Task5-MLLM-3D/train/CT-RATE-4791_processed \
    --with_template True \
    --image_size "32, 256, 256" \
    --model_max_length 768