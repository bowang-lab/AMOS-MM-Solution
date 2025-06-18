# FLARE 2025 3D MLLM Phi3 Baseline
This repository provides a baseline implementation for the FLARE 2025 3D multimodal medical image challenge.

# Installation
Requirements `Python >= 3.10.12` and `Python < 3.12` 
1. Make a python (or conda) virtual environment using: `python -m venv mllm` and activate it `source mllm/bin/activate`.
2. Clone the repo: `git clone https://github.com/bowang-lab/AMOS-MM-Solution.git` and `cd AMOS-MM-Solution`
3. Install requirements: `pip install -r requirements.txt`

# Training & Inference
We provide command line scripts for training on both tasks in the competition (medical report generation and visual question answering) and for doing inference with our post-processing technique.

## 1. Data Preperation
First clone the HuggingFace repo, where the dataset lives using:
`git clone https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task5-MLLM-3D`

Once that is done, we need to pre-process the data. You can do that by using the script:

`python Data/process/process_ct.py --json_in <PATH_TO_DATA_JSON> --nifti_dir <PATH_TO_DATA_DIR> --out_dir <OUTPUT_PATH> --workers <NUM_OF_WORKERS>`

This needs to be applied to both CT-RATE and AMOS datasets, as well as the validation dataset. If you cloned the HuggingFace dataset repo inside the main `AMOS-MM-Solution` directory, then the script for pre-processing the validation set would be:

`python process_ct.py --json_in FLARE-Task5-MLLM-3D/validation/val.json --nifti_dir FLARE-Task5-MLLM-3D/validation/images --out_dir FLARE-Task5-MLLM-3D/validation/val_processed`

## 2. Training 

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

To change the LLM used, you have to change the checkpoint path in HuggingFace using the arguemnt `model_name_or_path`.

The supported models are: `Phi3`, `Llama` famliy, `Gemma` famliy, `Qwen2`, and `Mistral`. If you change `model_name_or_path`, you have to also change `model_type` to the correct model type. 

The baseline uses LoRA fine-tuning for the LLM. You can disable that to fully fine-tune the model using `--lora_enable False`. If you want to freeze the LLM, there is an additional argument `--freeze_llm True`. 

## 3. Inference

NOTE: inference currently only supports Phi3. You need to manually change the model class to support other models.

To do inference for report generation, run the following command:
```
CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29500 infer.py \
  --model_name_or_path <PATH_TO_CHECKPOINT_DIR>   \
  --json_path <PATH_TO_VAL_JSON> \
  --data_root <PATH_TO_VAL_VOLUMES> \
  --model_max_length 768 \
  --prompt "simple" \
  --proj_out_num 256
```

After the scripts finishes running, this will generate 2 files at the same path where the model lives, provided in `--model_name_or_path`. The first file is `<NAME_OF_VAL_JSON>.csv`. This file contains the model generated reports and ground truth reports for each example, as well as GREEN score for each region. Any region mentioned in the ground truth report but not in model generated report is assigned a score of zero (false negative). Any region mentioned in the prediction but not the ground truth is compared with a normal ground truth, where the reference report becomes `f"{region} is normal."` The pipeline for validation can be found at `generate_green_score.py`.

To do VQA inference, run the following command:
```
CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29500 infer_vqa.py   \
  --model_name_or_path /path/to/trained/model \
  --json_path Data/AMOSMM.json \
  --image_size 32 256 256 \
  --model_max_length 512 \
  --proj_out_num 256
```

This will generate the `predictions.csv` file. This file will contain the global and local VQA predictions. NOTE: the local predictions need to be in a comma-separated format for each chain. For example, the answer for the following chain:

```
{
    "id": 1,
    "follow_up": -1,
    "question": "Is there evidence of cirrhosis in the liver?",
    "type": "finding_identification"
},
{
    "id": 2,
    "follow_up": 1,
    "question": "Which of the following features are present in the liver?",
    "type": "appearance_or_pattern",
    "choices": [
      "Small volume, uneven surface, disproportionate lobes, widened fissures",
      "Uniform size, smooth surface, normal fissures",
      "Enlarged volume, smooth surface, narrowed fissures",
      "None of the above"
     ]
}
```

Could be:
`Yes, Enlarged volume, smooth surface, narrowed fissures`.

## Results

The expected baseline resutls are:

Report Generation:

```
{
    "liver": 0.24076031746031748,
    "biliary system": 0.48416762931587704,
    "spleen": 0.5690602836879433,
    "pancreas": 0.5350826044703595,
    "kidneys": 0.22434423813734167,
    "gastrointestinal tract": 0.07610105074893808,
    "lymphatic system": 0.4945972495088409,
    "abdominal cavity and peritoneum": 0.31478494623655917,
    "endocrine system": 0.2296228710462287,
    "blood vessels": 0.10157232704402513,
    "musculoskeletal system": 0.4254729288975865,
    "lungs and pleura": 0.2141898823021273,
    "respiratory tract": 0.7048872180451128,
    "heart": 0.6663230240549824,
    "mediastinum": 0.5191605839416059,
    "esophagus": 0.6625850340136055,
    "breast tissue": 0.0,
    "diaphragm": 0.0
}
```

# Acknowledgements
* This codebase is built upon the M3D repository, so we gracefully acknowledge the authors for their work. 
