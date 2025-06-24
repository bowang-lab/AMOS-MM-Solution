#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Pre-process AMOS-MM")
    p.add_argument("--report_json", required=True, type=Path)
    p.add_argument("--vqa_json", required=True, type=Path)
    p.add_argument("--train_src", required=True, type=Path)
    p.add_argument("--val_src", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    return p.parse_args()

def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)

def case_id_from_image(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def preprocess_volume(task: tuple[str, str, Sequence[float], Sequence[int]]) -> None:
    raw_path, save_path, window, tgt_shape = task
    if Path(save_path).exists():
        return
    img = sitk.ReadImage(str(raw_path))
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 4:
        arr = arr[0]
    low, high = window
    arr = np.clip(arr.astype(np.float32), low, high)
    arr = (arr - low) / (high - low)
    arr = resize(arr, tgt_shape, anti_aliasing=True, preserve_range=True).astype(np.float32)
    arr = arr[np.newaxis, ...]
    _ensure_dir(Path(save_path).parent)
    np.save(save_path, arr)

def build_entry(case_id: str, spacing: Sequence[float], root: Path) -> Dict[str, Any]:
    base = root / "imagesProcessed" / case_id
    return {
        "image": str(base / f"{case_id}.npy"),
        "text": str(base / "report.json"),
        "vqa": str(base / "vqa.json"),
        "spacing": list(spacing),
        "impressions": str(base / "impressions.json"),
    }

def main() -> None:
    args = parse_args()
    report_meta = load_json(args.report_json)
    vqa_meta = load_json(args.vqa_json)
    processed_root = args.output
    window = (-160.0, 240.0)
    tgt_shape = (32, 256, 256)

    vqa_index: Dict[str, List[Any]] = {}
    
    for split in ("training", "validation"):
        for rec in vqa_meta[split]:

            file_name = Path(rec["image"]).name
            cid = case_id_from_image(file_name)
            vqa_index[cid] = rec["labels"]["qa"]
    
    output: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    tasks: List[tuple[str, str, Sequence[float], Sequence[int]]] = []
    
    for split_json, (split_out, src_dir) in {"training": ("train", args.train_src), "validation": ("val", args.val_src)}.items():
        for rec in report_meta.get(split_json, []):
            file_name = Path(rec["image"]).name
            raw_img = src_dir / file_name
            cid = case_id_from_image(file_name)
            spacing = rec["meta"].get("spacing", [])
            case_dir = processed_root / "imagesProcessed" / cid
            _ensure_dir(case_dir)
            save_npy = case_dir / f"{cid}.npy"
            tasks.append((str(raw_img), str(save_npy), window, tgt_shape))
            report_json_path = case_dir / "report.json"
            impressions_json_path = case_dir / "impressions.json"
            vqa_json_path = case_dir / "vqa.json"
            with report_json_path.open("w") as f:
                json.dump(rec["labels"]["report"]["findings"], f)
            with impressions_json_path.open("w") as f:
                json.dump(rec["labels"]["report"]["impression"], f)
            with vqa_json_path.open("w") as f:
                json.dump(vqa_index.get(cid, []), f)
            output[split_out].append(build_entry(cid, spacing, processed_root))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(preprocess_volume, tasks), total=len(tasks)))
    _ensure_dir(processed_root)
    with (processed_root / "splits.json").open("w") as f:
        json.dump(output, f, indent=4)
if __name__ == "__main__":
    main()
