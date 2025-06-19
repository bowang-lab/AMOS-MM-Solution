# FLARE 2025 3D MLLM Phi-3 Baseline

This repository provides a baseline implementation for the **FLARE 2025 3D Multimodal Medical Image Challenge**, including training and inference pipelines for both **medical report generation** and **visual question answering (VQA)** tasks.

---

## 📦 Installation

**Requirements:**  
- Python ≥ 3.10.12 and < 3.12

**Steps:**

1. Create and activate a virtual environment:
   ```bash
   python -m venv mllm && source mllm/bin/activate
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/bowang-lab/AMOS-MM-Solution.git
   cd AMOS-MM-Solution
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🗂️ Data Preparation

1. Clone the dataset repository:
   ```bash
   git clone https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task5-MLLM-3D
   ```

2. Preprocess each dataset using:
   ```bash
   python Data/process/process_ct.py \
     --json_in <PATH_TO_DATA_JSON> \
     --nifti_dir <PATH_TO_DATA_DIR> \
     --out_dir <OUTPUT_PATH> \
     --workers <NUM_WORKERS>
   ```

   For example, to preprocess the validation set:
   ```bash
   python process_ct.py \
     --json_in FLARE-Task5-MLLM-3D/validation/val.json \
     --nifti_dir FLARE-Task5-MLLM-3D/validation/images \
     --out_dir FLARE-Task5-MLLM-3D/validation/val_processed
   ```

---

## 🧠 Training

Train a baseline model with the following command:

```bash
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
  --save_strategy "steps" \
  --save_steps 20000 \
  --save_total_limit 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.0 \
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

### ✅ Notes:
- Vision backbone: [M3D ViT](https://github.com/BAAI-DCAI/M3D)
- Supported LLMs: `Phi3`, `LLaMA`, `Gemma`, `Qwen2`, `Mistral`
- Adjust `--model_name_or_path` and `--model_type` accordingly
- Use `--lora_enable False` to fully fine-tune the model
- Use `--freeze_llm True` to freeze the LLM

---

## 📄 Inference

> **Note:** Inference currently only supports **Phi-3**. To use other models, the model class must be manually modified.

### 📑 Report Generation

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29500 infer.py \
  --model_name_or_path <PATH_TO_CHECKPOINT_DIR> \
  --json_path <PATH_TO_VAL_JSON> \
  --data_root <PATH_TO_VAL_VOLUMES> \
  --model_max_length 768 \
  --prompt "simple" \
  --proj_out_num 256
```

Outputs:
- `<VAL_JSON_NAME>.csv`: Contains generated and ground-truth reports + GREEN score for each region.
- Scoring logic is in `generate_green_score.py`.

---

### ❓ VQA Inference

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29500 infer_vqa.py \
  --model_name_or_path <PATH_TO_CHECKPOINT_DIR> \
  --json_path Data/AMOSMM.json \
  --image_size 32 256 256 \
  --model_max_length 512 \
  --proj_out_num 256
```

Output:
- `predictions.csv`: Contains global and local VQA predictions.  
- For chains, answers should be comma-separated. For example:
  ```
  Yes, Enlarged volume, smooth surface, narrowed fissures
  ```

We use the script `eval_vqa.py` to evaluate predictions.

---

## 📊 Results

### 🔍 Report Generation

```json
{
  "liver": 0.2408,
  "biliary system": 0.4842,
  "spleen": 0.5691,
  "pancreas": 0.5351,
  "kidneys": 0.2243,
  "gastrointestinal tract": 0.0761,
  "lymphatic system": 0.4946,
  "abdominal cavity and peritoneum": 0.3148,
  "endocrine system": 0.2296,
  "blood vessels": 0.1016,
  "musculoskeletal system": 0.4255,
  "lungs and pleura": 0.2142,
  "respiratory tract": 0.7049,
  "heart": 0.6663,
  "mediastinum": 0.5192,
  "esophagus": 0.6626,
  "breast tissue": 0.0,
  "diaphragm": 0.0
}
```

### ❓ VQA

```json
{
  "global_accuracy": 0.1799,
  "local_accuracy": 0.5691
}
```

---

## 🙏 Acknowledgements

This baseline is built upon the excellent work in the [M3D repository](https://github.com/BAAI-DCAI/M3D). We thank the authors for making their code available.
