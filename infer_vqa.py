import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import SimpleITK as sitk
from skimage.transform import resize
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
import os
join = os.path.join
from LaMed.src.model.language_model import *
from collections import OrderedDict
import json
from tqdm import tqdm
import monai.transforms as mtf
from generate_green_score import GenerateGreenScore
import pandas as pd
import random




def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

# Set the seed for reproducibility
def main():
    parser = argparse.ArgumentParser(description='Script configuration')
    parser.add_argument('--is_val', type=bool, default=False, help='Validation flag')
    parser.add_argument('--image_size', type=int, nargs=3, default=(32, 256, 256), help='Image size as a tuple (C, H, W)')
    parser.add_argument('--lr_image_size', type=int, nargs=3, default=(32, 128, 128), help='Low resolution image size as a tuple (C, H, W)')
    parser.add_argument('--model_name_or_path', type=str, default='/scratch/ssd004/scratch/mohammed/results/hilt_64_320_1024', help='Model path or name')
    parser.add_argument('--json_path', type=str, default="/scratch/ssd004/scratch/mohammed/AMOSMM/AMOSMMVal.json", help='Path to JSON file')
    parser.add_argument('--model_max_length', type=int, default=768, help='Maximum model length')
    parser.add_argument('--proj_out_num', type=int, default=512, help='Project output number')
    parser.add_argument('--image_path', type=str, default="/scratch/ssd004/datasets/med-img-data/amosmm/ori_nii/imagesVa", help='Path to the image directory')
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--shuffle', type=bool, default=False)

    args = parser.parse_args()

    print("Arguments received:")
    print(f"is_val: {args.is_val}")
    print(f"image_size: {args.image_size}")
    print(f"lr_image_size: {args.lr_image_size}")
    print(f"model_name_or_path: {args.model_name_or_path}")
    print(f"json_path: {args.json_path}")
    print(f"model_max_length: {args.model_max_length}")
    print(f"proj_out_num: {args.proj_out_num}")
    print(f"image_path: {args.image_path}")
    print(f"prompt: {args.prompt}")
    print(f"shuffle: {args.shuffle}")

    seed_everything(42)

    device = torch.device('cuda') # 'cpu', 'cuda'
    dtype = torch.bfloat16 # or bfloat16, float16, float32

    image_size = args.image_size
    model_name_or_path = args.model_name_or_path
    json_path = args.json_path
    model_max_length = args.model_max_length
    proj_out_num = args.proj_out_num
    image_path = args.image_path
    prompt = args.prompt
    shuffle = args.shuffle

    with_acc = False
        
    transform = mtf.Compose(
            [
                mtf.Resize(image_size),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

    with open(json_path) as f:
        data = json.load(f)

    dataset = data['validation']

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
    model = model.to(device=device)
    template = True

    if not model.generation_config.pad_token_id:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    tag = json_path.split(os.sep)[-1].split(".")[0]
    if shuffle:
        path = model_name_or_path + os.sep + f'{tag}_shuffled.json'
    else:
        path = model_name_or_path + os.sep + f'{tag}.json'
    results = OrderedDict()
    results["validation"] = []

    correct, total = 0, 0    
    wrongs = []
    for item in tqdm(dataset):
        case_ = {}
        image_path = item['image']
        image_name = image_path.split(os.sep)[-1]

        case_["image"] = "./imagesTr/" + image_name + "nii.gz" 
        case_["labels"] = {"qa": []}

        ext = image_name.split(".")[-1]

        if ext == "npy":
            image = np.load(image_path)  # nomalized 0-1, C,D,H,W
            image = transform(image).unsqueeze(0).to(dtype=dtype, device=device)
        else:
            img_sitk = sitk.ReadImage(image_path)
            img_data = sitk.GetArrayFromImage(img_sitk)

            if len(img_data.shape) == 4:
                print(image_path, img_data.shape)
                img_data = img_data[1]

            img_data = np.clip(img_data, -160.0, 240.0)
            img_data = (img_data - np.min(img_data))/ (np.max(img_data) - np.min(img_data))
            img_data = np.expand_dims(img_data, 0)

            to_resize = mtf.Resize(image_size)
            to_tensor = mtf.ToTensor(dtype=dtype)
            image = to_tensor(to_resize(img_data)).unsqueeze(0).to(device=device)

            print(image_path, 'ori data shape:', img_data.shape, 'input tensor shape', image.shape)
            del img_sitk, img_data
        
        text_abs_path = item['vqa']
        with open(text_abs_path) as f:
            questions = json.load(f) # dict

        image_tokens = "<im_patch>" * proj_out_num * model.config.multipler
        for q_item in questions:
            
            case_q = {}

            question = q_item["question"]
            options = q_item["options"]
            
            case_q["question"] = question
            case_q["options"] = options
    
            choices = "Choices: A. {} B. {} C. {} D. {}".format(options["A"], options["B"], options["C"], options["D"])
            question = question + ' ' + choices

            if template:
                conversation = [{  
                    "role": "system", "content": "You are an AI assistant acting as a radiologist tasked with answering a multiple choice question based on a CT scan."},
                    {"role": "user", "content": image_tokens + ' ' + question}]
                input_txt = tokenizer.apply_chat_template(conversation, tokenize=False)
            else:
                input_txt = image_tokens + question
            input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
            
            generation = model.generate(image, input_id, max_new_tokens=10, do_sample=True, top_p=0.9, temperature=1.0)
            pred = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
            pred = pred.strip()

            case_q["prediction"] = pred[0]
            case_q["type"] = q_item["type"]
            
            if len(pred) == 0:
                pred = random.choice(["A", "B", "C", "D"]) 

            pred = pred[0]
            if pred not in ["A", "B", "C", "D"] :
                print(f"Incorrect option: {pred}")
                pred = random.choice(["A", "B", "C", "D"])

            if with_acc:
                answer = q_item["answer"]
                case_q["answer"] = answer
                if answer[0] == pred:
                    correct+=1
                else:
                    wrongs.append(case_q)
                total+=1
            case_["labels"]["qa"].append(case_q)

            if with_acc:
                print(str(correct / total))

        results["validation"].append(case_)

        with open(path, 'w') as f:
            json.dump(results, f)
        
        wrong_path = path.replace(".json", "_wrong.json")
        with open(wrong_path, 'w') as f:
            json.dump(wrongs, f)

    txt_path = path.replace("json", "txt")
    if with_acc:
        with open(txt_path, 'w') as file:
            # Convert the integer to a string and write it to the file
            file.write(str(correct / total))

if __name__ == '__main__':
    main()
