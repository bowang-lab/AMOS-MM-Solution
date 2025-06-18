import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')
from argparse import Namespace
import os
from LaMed.src.model.language_model import *
from tqdm import tqdm
from generate_green_score import GenerateGreenScore
import pandas as pd
from LaMed.src.dataset.multi_dataset import AMOSCapDataset

# Set the seed for reproducibility
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='Script configuration')
    parser.add_argument('--model_name_or_path', type=str, help='Model path or name')
    parser.add_argument('--model_max_length', type=int, default=768, help='Maximum model length')
    parser.add_argument('--proj_out_num', type=int, default=512, help='Project output number')
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--zoom', type=bool, default=False)
    parser.add_argument('--json_path', type=str, required=True, help='Val JSON file')
    parser.add_argument('--data_root', type=str, required=True, help='Val data root, where the volumes are')

    args = parser.parse_args()
    seed_everything(42)

    device = torch.device('cuda')
    dtype = torch.bfloat16  

    for key, value in vars(args).items():
        globals()[key] = value        
    
    model = LamedPhi3ForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map='auto',
        trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    model = model.eval()

    resize_size = model.config.image_size
    proj_out_num = args.proj_out_num * model.config.multipler

    args_dict = vars(args)
    print("Arguments received:")
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    tag = json_path.split(os.sep)[-1].split(".")[0]
    path = model_name_or_path + os.sep + f'{tag}.csv'

    # data_args = Namespace()
    # data_args.proj_out_num = proj_out_num
    # data_args.json_path = [json_path]
    # data_args.data_root = [data_root]
    # data_args.max_length = model_max_length
    # data_args.prompt = prompt
    # data_args.data_img_size = resize_size

    # dataset = AMOSCapDataset(data_args, tokenizer, mode='validation')

    # results = {
    #     'generated': [],
    #     'gt': [],
    #     'name': []
    # }

    # for item in tqdm(dataset):
    #     image_name = item["image_name"]

    #     image = item["image"].unsqueeze(0).to(device, dtype=dtype)
    #     input_id = item["input_id"].to(device)
    #     gt_text = item["answer"]

    #     generation = model.generate(image, input_id, max_new_tokens=512, do_sample=False, top_p=0.9, temperature=0)
    #     generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]

    #     results['gt'].append(gt_text)
    #     results['name'].append(image_name)
    #     results['generated'].append(generated_texts)

    #     results_df = pd.DataFrame(results)
    #     results_df.to_csv(path, index=False)
    
    print("Generating Green")
    g = GenerateGreenScore(path, cache_dir="./GREEN_model")
    results = g.run()

if __name__ == '__main__':
    main()