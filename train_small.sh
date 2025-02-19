#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=runs/run-log-%J.txt
#SBATCH --ntasks=10
#SBATCH -N 1
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --qos=a100_bowang

source /h/junma/.mvlm/bin/activate

# run "accelerate config" first!
JOB_ID=$SLURM_JOB_ID
PYTHONPATH=. accelerate launch --num_processes 1 --main_process_port 29510 LaMed/src/train/amos_train.py \
    --version v0 \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --cache_dir /scratch/ssd004/datasets/med-img-data/amosmm/LaMed/ \
    --model_type phi3 \
    --lora_enable True \
    --freeze_llm False \
    --lora_r 16 \
    --vision_tower vit3d \
    --pretrain_vision_model /scratch/ssd004/datasets/med-img-data/amosmm/LaMed/M3D-CLIP/pretrained_ViT.bin \
    --bf16 True \
    --output_dir  /scratch/ssd004/datasets/med-img-data/amosmm/trained/paper/phi3_150_i2f \
    --num_train_epochs 150 \
    --per_device_train_batch_size 4 \
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
    --task i2f \
    --json_path /fs01/home/junma/MedicalVLM/Data/AMOSMM.json \
    --data_root /scratch/ssd004/datasets/med-img-data/amosmm \
    --with_template True \
    --image_size "32, 256, 256" \
    --model_max_length 768