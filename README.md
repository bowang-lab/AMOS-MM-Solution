# AMOS-MM-Solution
This codebase is for our participation in the [MICCAI24 AMOS-MM: Abdominal Multimodal Analysis Challenge](https://www.codabench.org/competitions/3137/).

# Installation
Requirements `Python >= 3.10.12`
1. Make a python (or conda) virtual environment using: `python -m venv mllm` and activate it `source mllm/bin/activate`.
2. Clone the repo: `git clone https://github.com/bowang-lab/AMOS-MM-Solution.git` and `cd AMOS-MM-Solution`
3. Install requirements: `pip install -r requirements.txt`

# Training & Inference
We provide command line scripts for training on both tasks in the competition (medical report generation and visual question answering) and for doing inference with our post-processing technique.

## Data Preperation
To prepare the data, a json file needs to be made using the same structure as the one in `Data/AMOSMM.json`. After that is prepared, follow the steps below to trian the model and do inference.

## Training for MRG and VQA
After data is prepared, run the following command to train a Llama 3.1 model for report generation.

```
PYTHONPATH=. accelerate launch --num_processes 1 --main_process_port 29500 LaMed/src/train/amos_train.py \
    --version v0 \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --cache_dir "path/to/cache/dir" \
    --model_type llama \
    --freeze_llm True \
    --vision_tower vit3d \
    --pretrain_vision_model "path/to/vision/model" \
    --bf16 True \
    --output_dir "output/dir" \
    --num_train_epochs 100 \
    --per_device_train_batch_size 2 \
    --evaluation_strategy "no" \
    --do_eval False \
    --eval_accumulation_steps 1 \
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
    --dataloader_num_workers 4 \
    --report_to none \
    --prompt "simple" \
    --task mrg \
    --json_path "path/to/json" \
    --image_size "32, 256, 256" \
    --with_template True \
    --model_max_length 768
```
The argument `json_path` should point to the path of the json file we just prepared. Additionally, you have to set the `cache_dir` and `pretrain_vision_model`. For the vision model, we used the 3D ViT in [M3D](https://github.com/BAAI-DCAI/M3D). We provide additional arguments for this task like `zoom_in`, which uses organ segmentation masks to crop the abdomen based on a specific region (abdomen, chest, or pelvis), and `prompt` which controls the prompt. The "simple" prompt used can be found in `LaMed/src/dataset/prompts.py`.

To finetune your model for VQA instead of medical report generation, simple change the `task` argument to vqa. There are additional arguments for VQA, like `only_letter` and `with_reason`.

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
