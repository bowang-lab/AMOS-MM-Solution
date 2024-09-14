import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import SimpleITK as sitk
from skimage.transform import resize
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
from argparse import Namespace
import os
from accelerate import Accelerator
from LaMed.src.model.language_model import *
import json
from tqdm import tqdm
import monai.transforms as mtf
from generate_green_score import GenerateGreenScore
import pandas as pd
from LaMed.src.dataset.multi_dataset import prompt_templates
import re
from LaMed.src.dataset.utils import read_numpy_or_dicom
from utils.postprocessor import PostProcessor
from LaMed.src.dataset.multi_dataset import AMOSCapDataset

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

from torch.utils.data.dataloader import default_collate
def custom_collate(batch):    

    images, input_ids, answer, image_name = tuple(
        [b[key] for b in batch] for key in ('image', 'input_id', 'answer', 'image_name'))

    images_ = {i: a for i, a in enumerate(images)}
    input_ids = {i: a for i, a in enumerate(input_ids)}
    answers_ = {i: a for i, a in enumerate(answer)}
    image_name_ = {i: a for i, a in enumerate(image_name)}

    return_dict = dict(
        image=images_,
        input_id=input_ids,
        answer=answers_,
        image_name=image_name_
    )

    return return_dict

# Set the seed for reproducibility
def main():
    parser = argparse.ArgumentParser(description='Script configuration')
    parser.add_argument('--is_val', type=bool, default=False, help='Validation flag')
    parser.add_argument('--model_name_or_path', type=str, default='/scratch/ssd004/scratch/mohammed/results/hilt_64_320_1024', help='Model path or name')
    parser.add_argument('--json_path', type=str, default="/scratch/ssd004/scratch/mohammed/AMOSMM/AMOSMMVal.json", help='Path to JSON file')
    parser.add_argument('--model_max_length', type=int, default=768, help='Maximum model length')
    parser.add_argument('--proj_out_num', type=int, default=512, help='Project output number')
    parser.add_argument('--image_path', type=str, default="/scratch/ssd004/datasets/med-img-data/amosmm/ori_nii/imagesVa", help='Path to the image directory')
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--organs', metavar='N', type=str, nargs='+', default=["abdomen", "pelvis", "chest"])
    parser.add_argument('--zoom', type=bool, default=False)
    parser.add_argument('--post_process', metavar='N', type=str, nargs='+', default=[])

    args = parser.parse_args()

    seed_everything(42)

    device = torch.device('cuda')
    dtype = torch.bfloat16  # or bfloat16, float16, float32
    with_template = True
    green = False   

    for key, value in vars(args).items():
        globals()[key] = value        
    
    if "llama" in model_name_or_path:
        model = LamedLlamaForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir='/scratch/ssd004/datasets/med-img-data/amosmm/trained/cache/',
            torch_dtype=dtype,
            device_map='auto',
            trust_remote_code=True)
    elif "gemma" in model_name_or_path:
        model = LamedGemmaForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir='/scratch/ssd004/datasets/med-img-data/amosmm/trained/cache/',
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map='auto')
    else:
        model = LamedPhi3ForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir='/scratch/ssd004/datasets/med-img-data/amosmm/trained/cache/',
            torch_dtype=dtype,
            device_map='auto',
            trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir='/scratch/ssd004/datasets/med-img-data/amosmm/trained/cache/',
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    model = model.eval()

    if model.config.any_res_image_size:
        resize_size = model.config.any_res_image_size
    else:
        resize_size = model.config.image_size

    proj_out_num = args.proj_out_num * model.config.multipler

    args_dict = vars(args)
    print("Arguments received:")
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    tag = json_path.split(os.sep)[-1].split(".")[0]
    path = model_name_or_path + os.sep + f'{tag}.csv'

    if os.path.exists(path):
        results = pd.read_csv(path)
        results = results.to_dict(orient='list')
    else:
        results = OrderedDict()
        results['names'] = []
        for organ in organs:
            results[f'generated-{organ}'] = []
            results[f'gt-{organ}'] = []

    data_args = Namespace()
    data_args.proj_out_num = proj_out_num
    data_args.json_path = json_path
    data_args.data_root = ""
    data_args.max_length = model_max_length
    data_args.prompt = prompt
    data_args.zoom_in = zoom
    data_args.organs = organs
    data_args.with_seg_mask = False
    data_args.with_template= with_template
    data_args.data_img_size = resize_size

    dataset = AMOSCapDataset(data_args, tokenizer, mode='validation')

    for item in tqdm(dataset):
        image_name = item["image_name"]

        organs_ = ["abdomen", "pelvis", "chest"]
        if green:
            organs_ = item["answer"].keys()

        for organ in organs_:
            
            image = item["image"][organ].unsqueeze(0).to(device, dtype=dtype)
            input_id = item["input_id"][organ].to(device)
            generation = model.generate(image, input_id, segs=None, max_new_tokens=512, do_sample=False, top_p=0.9, temperature=1)
            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
            results['generated-' + str(organ)].append(generated_texts[0])

            if green:
                gt_text = item["answer"][organ]
                results['gt-' + str(organ)].append(gt_text)
            else:
                results['gt-' + str(organ)].append("")

        missing_organs = [o for o in organs if o not in organs_]
        for m_organ in missing_organs:
            results['gt-' + str(m_organ)].append("")
            results['generated-' + str(m_organ)].append("")

        results['names'].append(image_name)
        results_df = pd.DataFrame(results)
        results_df.to_csv(path, index=False)
    
    if len(post_process) > 0:
        print("Using post processing methods:", post_process)
        pp = PostProcessor(results, post_process, model, tokenizer, dataset)
        results = pp.run()
        results_df = pd.DataFrame(results)
        results_df.to_csv(path, index=False)

    if green:
        print("Generating Green")
        g = GenerateGreenScore(path, cache_dir="/scratch/ssd004/datasets/med-img-data/amosmm/green", organs=organs)
        results = g.run()

if __name__ == '__main__':
    main()