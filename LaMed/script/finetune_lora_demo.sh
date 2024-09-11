#!/bin/bash

# run "accelerate config" first!

accelerate launch LaMed/src/train/train.py \
    --version v0 \
    --model_name_or_path ./LaMed/M3D-LaMed-Phi-3-4B \
    --model_type phi3 \
    --lora_enable True \
    --vision_tower vit3d \
    --pretrain_vision_model LaMed/M3D-CLIP/pretrained_ViT.bin \
    --bf16 True \
    --output_dir ./LaMed/output/LaMed-finetune-phi3-demo \
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
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
    --report_to tensorboard