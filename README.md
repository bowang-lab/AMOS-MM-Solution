# FLARE 2025 3D MLLM Phi3 Baseline
This repository provides a baseline implementation for the FLARE 2025 3D multimodal medical image challenge.

# Installation
Requirements `Python >= 3.10.12` and `Python < 3.12` 
1. Make a python (or conda) virtual environment using: `python -m venv mllm` and activate it `source mllm/bin/activate`.
2. Clone the repo: `git clone https://github.com/bowang-lab/AMOS-MM-Solution.git` and `cd AMOS-MM-Solution`
3. Install requirements: `pip install -r requirements.txt`

# Training & Inference
We provide command line scripts for training on both tasks in the competition (medical report generation and visual question answering) and for doing inference with our post-processing technique.

## Data Preperation
First clone the HuggingFace repo, where the dataset lives using:
`git clone https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task5-MLLM-3D`

Once that is done, we need to pre-process the data. You can do that by using the script:

`python Data/process/process_ct.py --json_in <PATH_TO_DATA_JSON> --nifti_dir <PATH_TO_DATA_DIR> --out_dir <OUTPUT_PATH> --workers <NUM_OF_WORKERS>`

This needs to be applied to both CT-RATE and AMOS datasets, as well as the validation dataset. If you cloned the repo inside the main `AMOS-MM-Solution` dir, then the script would be:

`python process_ct.py --json_in FLARE-Task5-MLLM-3D/validation/val.json --nifti_dir FLARE-Task5-MLLM-3D/validation/images --out_dir FLARE-Task5-MLLM-3D/validation/val_processed`

for pre-processing the validation set.

## Training 

Once pre-processing is done, you can train a baseline model using:

```
PYTHONPATH=. accelerate launch --num_processes 1 --main_process_port 29500 LaMed/src/train/amos_train.py \
    --version v0 \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --cache_dir <CACHE_DIR> \
    --model_type phi3 \
    --lora_enable True \
    --lora_r 16 \
    --vision_tower vit3d \
    --pretrain_vision_model <VIT_PATH> \
    --bf16 True \
    --output_dir results/baseline \
    --num_train_epochs 75 \
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
    --task all \
    --json_path <PATH_TO_AMOS_JSON> <PATH_TO_CT-RATE_JSON> \
    --data_root <PATH_TO_AMOS_VOLUMES> <PATH_TO_CT-RATE_VOLUMES> \
    --with_template True \
    --image_size "32, 256, 256" \
    --model_max_length 1024
```
For the vision model, we used the 3D ViT in [M3D](https://github.com/BAAI-DCAI/M3D). 


## Inference
To do inference for MRG, run the following command:
```
CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29500 infer.py \
  --model_name_or_path /path/to/trained/model \
  --json_path Data/AMOSMM.json \
  --model_max_length 768 \
  --prompt "simple" \
  --post_process "normality" "focused_inference" \
  --proj_out_num 256
```
The argument `post_process` adds two additional steps when inference on the model is done. The first is a knowledge-base normality finding, and the second is a focused inference based on specified questions. You can find the knowledge base at `utils/postprocessor.py`. The ones currently used, especially for the focused inference, are specific to the competition dataset and our submissions, and should be changed depending on the usecase.

To do VQA inference, run the following command:
```
CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29500 infer_vqa.py   \
  --model_name_or_path /path/to/trained/model \
  --json_path Data/AMOSMM.json \
  --image_size 32 256 256 \
  --model_max_length 512 \
  --proj_out_num 256
```
An additional argument `with_acc` is used to control whether to also calculate the VQA accuracy. You need to have the correct answers in the same format as the competiton for this to work. 

# Acknowledgements
* We highly appreciate all the challenge organizers of the MICCAI24 AMOS-MM challenge.
* This codebase is built upon the M3D repository, so we gracefully acknowledge the authors for their work. 
