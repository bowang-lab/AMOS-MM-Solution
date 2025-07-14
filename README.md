# AMOS-MM-Solution

This repository contains our solution for the [MICCAI24 AMOS-MM: Abdominal Multimodal Analysis Challenge](https://www.codabench.org/competitions/3137/).

---

## Installation

**Requirements:**  
- Python ≥ 3.10.12 and < 3.12

**Setup steps:**

1. Create a Python (or conda) virtual environment:

    ```bash
    python -m venv mllm
    source mllm/bin/activate
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

## Dataset Download

To replicate or expand upon our experiments, download the AMOS-MM dataset from [here](https://era-ai-biomed.github.io/amos/dataset.html#download). Once downloaded, you can proceed with dataset preparation.

---

## Data Preparation

The dataset requires a JSON file structured similarly to `Data/dataset.json`. To generate it, run the following command:

```bash
python prepare_data.py \
  --report_json <PATH_TO_report_generation_train_val.json> \
  --vqa_json <PATH_TO_vqa_train_val.json> \
  --output <PATH_TO_OUTPUT_DIR> \
  --train_src <PATH_TO_imagesTr> \
  --val_src <PATH_TO_imagesVa>
```

---

## Training

### Medical Report Generation (MRG)

Once data preparation is complete, train the LLaMA 3.1 model for report generation using:

```bash
PYTHONPATH=. accelerate launch --num_processes 1 --main_process_port 29500 LaMed/src/train/amos_train.py \
    --version v0 \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --cache_dir <WHERE_MODEL_WILL_BE_SAVED> \
    --model_type llama \
    --freeze_llm True \
    --vision_tower vit3d \
    --pretrain_vision_model <PATH_TO_PRETRAINED_VISION_MODEL> \
    --bf16 True \
    --output_dir <WHERE_TO_SAVE_MODEL> \
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
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    --report_to none \
    --prompt "simple" \
    --task mrg \
    --json_path <PATH_TO_DATASET_JSON> \
    --image_size "32, 256, 256" \
    --with_template True \
    --model_max_length 768
```

- The `json_path` should point to the JSON file prepared earlier.
- Set `cache_dir` and `pretrain_vision_model` appropriately.
- The vision model we used is the 3D ViT from [M3D](https://github.com/BAAI-DCAI/M3D).
- Additional arguments:
  - `zoom_in`: uses organ segmentation masks for region cropping.
  - `prompt`: controls the prompt format (e.g. `"simple"` in `LaMed/src/dataset/prompts.py`).

---

### Visual Question Answering (VQA)

To fine-tune the model for VQA, change the `--task` argument to `vqa`. Additional arguments include:
- `only_letter`: to restrict answers to single letters.
- `with_reason`: to include reasoning in answers.

---

## Optional: Training the Triplet Model

For Binary-based Questioning (BQ), first prepare triplets:

```bash
python scripts/triplet_extraction.py \
  --json_path <PATH_TO_DATASET_JSON> \
  --openai_key <OPEN_AI_KEY>
```

- You can modify the model used for triplet extraction inside the script.
- The triplet files will be named to align with the report files for seamless training.

To train the triplet model, use the same training command as above, adding:

```
--triplet True
```

---

## Inference

### MRG Inference

Run inference for medical report generation:

```bash
CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29500 infer.py \
  --model_name_or_path <PATH_TO_TRAINED_MODEL> \
  --json_path <PATH_TO_DATA_JSON> \
  --model_max_length 768 \
  --prompt "simple" \
  --post_process "normality" "bq" \
  --triplet_model_path <PATH_TO_TRAINED_TRIPLET_MODEL> \
  --proj_out_num 256
```

**Note:**  
- If you did not train a triplet model, omit the `"bq"` argument and `--triplet_model_path`.
- The `post_process` argument enables:
  - Knowledge-based normality inference.
  - Focused questioning based on specific findings.
- The knowledge base is defined in `utils/postprocessor.py`. Adapt it for different datasets.

---

### VQA Inference

Run VQA inference with:

```bash
CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29500 infer_vqa.py \
  --model_name_or_path <PATH_TO_TRAINED_MODEL> \
  --json_path <PATH_TO_DATA_JSON> \
  --model_max_length 512 \
  --proj_out_num 256
```

- The optional `--with_acc` argument computes VQA accuracy if ground truth answers are available in the competition format.

---

## Editing the Knowledge Base for NN and BQ

Our paper introduces two report augmentation methods:
- **Naive Normality (NN)**
- **Binary-based Questioning (BQ)**

Both methods rely on a pre-defined knowledge base specific to AMOS-MM. To customize this for other datasets, edit the mappings in:

```
utils/postprocessor.py
```

---

## Acknowledgements

- We thank the organizers of the MICCAI24 AMOS-MM challenge for their efforts.
- This codebase builds upon the [M3D repository](https://github.com/BAAI-DCAI/M3D), and we gratefully acknowledge its authors.

## Bibtex
```
@article{baharoon2025exploring,
  title={Exploring the Design Space of 3D MLLMs for CT Report Generation},
  author={Baharoon, Mohammed and Ma, Jun and Fang, Congyu and Toma, Augustin and Wang, Bo},
  journal={arXiv preprint arXiv:2506.21535},
  year={2025}
}
```
