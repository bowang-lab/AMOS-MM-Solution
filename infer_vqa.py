#!/usr/bin/env python3
import argparse, json, os, random, warnings, re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import monai.transforms as mtf
from tqdm import tqdm
from transformers import AutoTokenizer

from LaMed.src.model.language_model import LamedPhi3ForCausalLM
from LaMed.src.dataset.multi_dataset import read_image

warnings.filterwarnings("ignore")
join = os.path.join

def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_pred(pred: str) -> str:
    lines = [l.strip() for l in pred.splitlines() if l.strip()]
    numbered = [
        re.sub(r"^\s*\d+\.\s*", "", l).strip()
        for l in lines
        if re.match(r"^\s*\d+\.\s*", l)
    ]
    return ", ".join(numbered) if numbered and len(numbered) == len(lines) else pred

def build_prompt(q_txt: str, proj_tokens: int, tokenizer) -> str:
    """Exact same chat-template wrapper used during training."""
    image_tokens = "<im_patch>" * proj_tokens
    conversation = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant acting as a radiologist tasked with "
                "answering a multiple-choice question based on a CT scan."
            ),
        },
        {"role": "user", "content": image_tokens + " " + q_txt},
    ]
    return tokenizer.apply_chat_template(conversation, tokenize=False)

def main():
    p = argparse.ArgumentParser("AMOS-VQA inference (identical prompt setup)")
    p.add_argument("--model_name_or_path")
    p.add_argument("--json_path", required=True, help="Validation JSON file")
    p.add_argument("--data_root", required=True, help="Folder with volumes")
    p.add_argument("--model_max_length", type=int, default=768)
    p.add_argument("--proj_out_num", type=int, default=256)
    args = p.parse_args()

    seed_everything()
    device, dtype = torch.device("cuda"), torch.bfloat16

    with open(args.json_path) as f:
        dataset = json.load(f)

    model = LamedPhi3ForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    resize_size = model.config.image_size
    transform = mtf.Compose([mtf.Resize(resize_size), mtf.ToTensor(dtype=torch.float)])
    proj_tokens = args.proj_out_num * model.config.multipler

    rows = []

    for sample in tqdm(dataset):
        case_id = sample["case_id"]
        img_path = join(args.data_root, case_id)
        image = transform(read_image(img_path)).unsqueeze(0).to(device, dtype)

        for g in sample["global_vqa"]:
            q_txt = g["question"].rstrip()
            if "choices" in g and g["choices"]:
                q_txt = f"{q_txt} Choices: {g['choices']}"

            prompt = build_prompt(q_txt, proj_tokens, tokenizer)
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

            out = model.generate(
                image, input_ids,
                max_new_tokens=128, do_sample=True, top_p=0.9, temperature=1.0
            )
            pred = tokenizer.decode(out[0], skip_special_tokens=True).replace(prompt, "").strip()

            rows.append(dict(
                case_id=case_id,
                scope="global",
                question_id="",
                question=g["question"],
                prediction=pred or "None",
            ))

        locals_ = sample["local_vqa"]
        roots = [q for q in locals_ if q["follow_up"] == -1]
        id2qs = {}
        for q in locals_:
            id2qs.setdefault(q["follow_up"], []).append(q)

        for root in roots:
            chain = [root] + id2qs.get(root["id"], [])

            q_lines = []
            for idx, q in enumerate(chain, 1):
                q_line = q["question"].rstrip()
                if "choices" in q and q["choices"]:
                    q_line = f"{q_line} Choices: {q['choices']}"
                q_lines.append(f"{idx}. {q_line}")
            q_txt = "\n".join(q_lines)

            prompt = build_prompt(q_txt, proj_tokens, tokenizer)
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

            out = model.generate(
                image, input_ids,
                max_new_tokens=36, do_sample=True, top_p=0.9, temperature=1.0
            )
            pred = tokenizer.decode(out[0], skip_special_tokens=True).replace(prompt, "").strip()
            pred = format_pred(pred)

            rows.append(dict(
                case_id=case_id,
                scope="local",
                question_id=root["id"],
                question=q_txt,
                prediction=pred or "None",
            ))

    out_csv = Path(args.model_name_or_path) / "predictions.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

if __name__ == "__main__":
    main()