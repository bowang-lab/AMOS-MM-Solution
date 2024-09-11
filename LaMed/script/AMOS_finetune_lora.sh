#!/bin/bash

# run "accelerate config" first!
JOB_ID=$SLURM_JOB_ID

# CUDA_VISIBLE_DEVICES="0" PYTHONPATH=. accelerate launch
PYTHONPATH=. torchrun --nproc_per_node=1 LaMed/src/train/amos_train.py \
    --version v0 \
    --model_name_or_path /scratch/ssd004/scratch/mohammed/models/M3D-LaMed-Phi-3-4B \
    --model_type phi3 \
    --lora_enable True \
    --vision_tower vit3d \
    --pretrain_vision_model /scratch/ssd004/scratch/mohammed/models/M3D-CLIP/pretrained_ViT.bin \
    --bf16 True \
    --output_dir /checkpoint/mohammed/${JOB_ID}/results_test/ \
    --num_train_epochs 100 \
    --per_device_train_batch_size 1 \
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
    --report_to tensorboard \
    --use_hilt False \
    --task vqa \
    --json_path /scratch/ssd004/scratch/mohammed/AMOSMM/AMOSMMTraining.json \
    --data_root /scratch/ssd004/datasets/med-img-data/amosmm \
    --image_size "64, 320, 320" \
    --lr_image_size "32, 256, 256" \
    --model_max_length 512

    # /scratch/ssd004/scratch/mohammed/AMOSMM/AMOSMMTraining.json