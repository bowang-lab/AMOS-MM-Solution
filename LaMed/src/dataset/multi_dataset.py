import random
import os
join = os.path.join
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision

import re
import json
import pandas as pd

import SimpleITK as sitk
import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta

from .utils import read_numpy_or_dicom, REGION_TO_ORGAN_IDS_MAPPING
from ..utils.utils import mask2box
from .dataset_info import dataset_info
from .prompt_templates import Caption_templates, PosREC_templates, PosREG_templates, Seg_templates
from .term_dictionary import term_dict
from .prompts import prompt_templates
from skimage.transform import resize
import pdb


def read_image(path):
    ext = path.split(os.sep)[-1].split(".")[-1]
    if ext == "npy":
        image = np.load(path)  # nomalized 0-1, C,D,H,W
    else:
        img_sitk = sitk.ReadImage(path)
        img_data = sitk.GetArrayFromImage(img_sitk)

        if len(img_data.shape) == 4:
            img_data = img_data[1]

        img_data = np.clip(img_data, -160.0, 240.0)
        img_data = (img_data - np.min(img_data))/ (np.max(img_data) - np.min(img_data))
        image = np.expand_dims(img_data, 0)
        del img_data, img_sitk
    return image

class AMOSCapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        print("Arguments provided in 'args':")
        for key, value in vars(self.args).items():
            print(f"{key}: {value}")

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train_val":
            with open(args.json_path, 'r') as file:
                self.json_file = json.load(file)
            self.data_list = self.json_file["train"]
            with open(args.json_path.replace("Training", "Val"), 'r') as file:
                json_file_val = json.load(file)
            self.data_list.extend(json_file_val["validation"])
        else:
            with open(args.json_path, 'r') as file:
                self.json_file = json.load(file)
            self.data_list = self.json_file[mode]
        self.data_list = self.data_list[:3]
        print(f"Length dataset: {len(self.data_list)}")

        if args.prompt in prompt_templates.keys():               
            print("Prompt is a dict")
            self.prompt = prompt_templates[args.prompt]
        else:
            self.prompt = args.prompt

        self.caption_prompts = [
            "abdomen",
            "chest",
            "pelvis",
        ]

        train_transform = mtf.Compose([
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        self.resize_transform = mtf.Resize(args.data_img_size)
        set_track_meta(False)

        if mode == 'train' or mode == "train_val":
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
            self.return_all = True
        elif 'test' in mode:
            self.transform = val_transform

    def _to_uniform_depth(self, img, spacing, to_spacing=5):
        current_shape = img.shape
        current_spacing = spacing[2]
        if current_spacing != to_spacing:            
            scale_by = current_spacing / to_spacing
            to_shape = (int(current_shape[1] * scale_by), current_shape[2], current_shape[3])
            img = torch.nn.functional.interpolate(img.unsqueeze(0), to_shape).squeeze(0)
        return img
    
    def _zoom_in(self, img, mask_path, organ):

        # read mask
        image_name = mask_path.split(os.sep)[-1]

        ext = image_name.split(".")[-1]
        seg = read_numpy_or_dicom(mask_path, ext)
        _, DI, HI, WI = img.shape
        seg = resize(seg, (DI, HI, WI), order=0, preserve_range=True, anti_aliasing=False).astype(seg.dtype)
        seg[~np.isin(seg, REGION_TO_ORGAN_IDS_MAPPING[organ])] = 0
        nonzero_coords = np.argwhere(seg > 0)
        min_coords = nonzero_coords.min(axis=0)
        max_coords = nonzero_coords.max(axis=0)
        z_min, y_min, x_min = min_coords
        z_max, y_max, x_max = max_coords

        cropped_img = img[:, z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        del seg
        return cropped_img
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem_validation__(self, idx):
        data = self.data_list[idx]
        try:
            text_path = data["text"]
            if text_path.startswith('/'):
                text_abs_path = text_path
            else:
                text_abs_path = os.path.join(self.data_root, text_path)
            
            with open(text_abs_path) as f:
                raw_text = json.load(f) # dict
                while "findings" in raw_text.keys():
                    raw_text = raw_text["findings"]
                raw_text = {k: v for k, v in raw_text.items() if k in self.args.organs}
        except:
            raw_text = None
        
        questions = {}
        for organ in self.args.organs:
            if isinstance(self.prompt, dict):
                prompt = self.prompt[organ]
                if self.args.with_template:
                    msg = prompt
                    system = msg.split(".")[0] + "."
                    content = ".".join(msg.split(".")[1:])

                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": content}
                    ]
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
                    questions[organ] = self.image_tokens + prompt
                else:
                    questions[organ] = self.image_tokens + prompt
            else: 
                questions[organ] = self.image_tokens + self.prompt + organ
        
        image_path = data["image"]
        if image_path.startswith('/'):
            image_abs_path = image_path 
        else:
            image_abs_path = os.path.join(self.data_root, image_path)

        images = {}
        for organ in self.args.organs:
            image = read_image(image_abs_path)
            image = self.transform(image)
            image = self._to_uniform_depth(image, data["spacing"], to_spacing=5)
            if self.args.zoom_in:
                image = self._zoom_in(image, data["mask"], organ)
            image = self.resize_transform(image)
            images[organ] = image

        input_ids = {}
        for organ in self.args.organs:
            question = questions[organ]
            
            text_tensor = self.tokenizer(
                question, return_tensors="pt"
            )
            input_id = text_tensor["input_ids"]
            input_ids[organ] = input_id

        ret = {
            'image': images,
            'input_id': input_ids,
            'question': question,
            'answer': raw_text,
            'image_name': image_path.split(os.sep)[-1],
            'question_type': "Caption",
        }

        return ret

    def __getitem__(self, idx):
        if self.mode == "validation":
            return self.__getitem_validation__(idx)
        
        data = self.data_list[idx]
        text_path = data["text"]
        if text_path.startswith('/'):
            text_abs_path = text_path
        else:
            text_abs_path = os.path.join(self.data_root, text_path)
        
        if text_abs_path.endswith('.json'):
            with open(text_abs_path) as f:
                raw_text = json.load(f)
                while "findings" in raw_text.keys():
                    raw_text = raw_text["findings"]

                raw_text = {k: v for k, v in raw_text.items() if k in self.args.organs}
                if len(raw_text.keys()) == 3:
                    organ = random.choices(["abdomen", "chest", "pelvis"], weights=[0.1, 0.7, 0.2], k=1)[0]
                elif len(raw_text.keys()) == 2:
                    organ = random.choices(list(raw_text.keys()), weights=[0.50, 0.50], k=1)[0]
                else:
                    organ = list(raw_text.keys())[0]
                answer = raw_text[organ]
        else:
            print(f"text Error in __getitem__ at index {idx}: {e}, file suffix should be .txt or .json")
        
        if self.args.with_impressions:
            with open(data["impressions"]) as f:
                raw_text = json.load(f)
                raw_text = {k: v for k, v in raw_text.items() if k in self.args.organs}
                impressions = " ".join(raw_text[organ])
                answer = f"<FINDINGS>{answer}<FINDINGS><IMPRESSIONS>{impressions}<IMPRESSIONS>"
        
        if isinstance(self.prompt, dict):
            prompt = self.prompt[organ]
            if self.args.with_template:
                msg = prompt
                system = msg.split(".")[0] + "."
                content = ".".join(msg.split(".")[1:])
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content}
                ]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            question = self.image_tokens + prompt
        else: 
            question = self.image_tokens + self.prompt + organ

        image_path = data["image"]
        if image_path.startswith('/'):
            image_abs_path = image_path 
        else:
            image_abs_path = os.path.join(self.data_root, image_path)

        image = read_image(image_abs_path)
        image = self.transform(image)
        image = self._to_uniform_depth(image, data["spacing"], to_spacing=5)
        if self.args.zoom_in:
            image = self._zoom_in(image, data["mask"], organ)

        image = self.resize_transform(image)
        seg_mask = None
        if self.args.with_seg_mask:
            _, DI, HI, WI = image.shape
            ext = data["mask"].split(os.sep)[-1].split(".")[-1]
            seg_mask = read_numpy_or_dicom(data["mask"], ext)
            seg_mask = self.transform(resize(seg_mask, (DI, HI, WI), anti_aliasing=False)).unsqueeze(0)

        text_tensor = self.tokenizer(
            question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )

        input_id = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]

        valid_len = torch.sum(attention_mask)
        if valid_len < len(input_id):
            input_id[valid_len] = self.tokenizer.eos_token_id

        question_tensor = self.tokenizer(
            question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )

        question_len = torch.sum(question_tensor["attention_mask"][0])

        label = input_id.clone()
        label[:question_len] = -100
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            label[label == self.tokenizer.pad_token_id] = -100
            if valid_len < len(label):
                label[valid_len] = self.tokenizer.eos_token_id
        else:
            label[label == self.tokenizer.pad_token_id] = -100

        ret = {
            'image': image,
            'input_id': input_id,
            'label': label,
            'attention_mask': attention_mask,
            'segs': seg_mask,
            'question': question,
            'answer': answer,
            'question_type': "Caption",
        }

        return ret

class AMOSVQADataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        
        print("Arguments provided in 'args':")
        for key, value in vars(self.args).items():
            print(f"{key}: {value}")

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train_val":
            with open(args.json_path, 'r') as file:
                self.json_file = json.load(file)
            self.data_list = self.json_file["train"]
            with open(args.json_path.replace("Training", "Val"), 'r') as file:
                json_file_val = json.load(file)
            self.data_list.extend(json_file_val["validation"])
        else:
            with open(args.json_path, 'r') as file:
                self.json_file = json.load(file)
            self.data_list = self.json_file[mode]
        print(f"Length dataset: {len(self.data_list)}")

        train_transform = mtf.Compose(
            [
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
                mtf.Resize(args.data_img_size),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                    mtf.Resize(args.data_img_size),
                ]
            )
        set_track_meta(False)

        if mode == 'train' or mode == "train_val":
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_tries = 10
        for _ in range(max_tries):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                if image_path.startswith('/'):
                    image_abs_path = image_path 
                else:
                    image_abs_path = os.path.join(self.data_root, image_path)

                image = read_image(image_abs_path)
                image = self.transform(image)

                vqa_path = data["vqa"]
                if vqa_path.startswith('/'):
                    vqa_abs_path = vqa_path
                else:
                    vqa_abs_path = os.path.join(self.data_root, vqa_path)
                
                if vqa_abs_path.endswith('.json'):
                    with open(vqa_abs_path) as f:
                        qs = json.load(f) # dict
                        if self.args.with_gen:
                            vqa_abs_path_ = vqa_abs_path.replace(".json", "_gpt4-o.json")
                            if os.path.exists(vqa_abs_path_):
                                with open(vqa_abs_path_) as f_:
                                    qs.extend(json.load(f_))
                        vqa_data = random.choices(qs, k=1)[0]
                        options = vqa_data["options"]
                        question = vqa_data["question"]
                        choices = "Choices: A. {} B. {} C. {} D. {}".format(options["A"], options["B"], options["C"], options["D"])
                        question = question + ' ' + choices
                        if self.args.only_letter:
                            answer = f"{vqa_data['answer']}."
                        else:
                            answer = "{}. {}".format(vqa_data["answer"], options[vqa_data["answer"]])
                        if self.args.with_reason:
                            answer += f". The reason for choosing answer {vqa_data['answer']} is: {vqa_data['reasoning']}"
                        if self.args.with_report:
                            with open(data["text"]) as f:
                                report = json.load(f)
                                report_string = "<FINDINGS>"
                                for k, v in report.items():
                                    report_string += k + ": " + v
                                report_string += "<FINDINGS>"
                            with open(data["impressions"]) as f:
                                report = json.load(f)
                                report_string += "<IMPRESSIONS>"
                                for k, v in report.items():
                                    report_string += k + ": " + " ".join(v)
                                report_string += "<IMPRESSIONS>"
                            answer += report_string
                else:
                    print(f"text Error in __getitem__ at index {idx}, file suffix should be .txt or .json")

                if self.args.with_template:
                    conversation = [{  
                        "role": "system", "content": "You are an AI assistant acting as a radiologist tasked with answering a multiple choice question based on a CT scan."},
                        {"role": "user", "content": self.image_tokens + ' ' + question}]
                    question = self.tokenizer.apply_chat_template(conversation, tokenize=False)
                else:
                    question = self.image_tokens + ' ' + question

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt",
                )
                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'segs': None,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                }

                return ret
            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}, name: {self.data_list[idx]}")
                idx = random.randint(0, len(self.data_list) - 1)


class UniDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train', **kwargs):
        super(UniDatasets, self).__init__()
        cap_kwargs = {k: v for k, v in kwargs.items() if k in AMOSCapDataset.__init__.__code__.co_varnames}
        vqa_kwargs = {k: v for k, v in kwargs.items() if k in AMOSVQADataset.__init__.__code__.co_varnames}
        self.ds_list = [
            AMOSCapDataset(args, tokenizer, mode=mode, **cap_kwargs),
            AMOSVQADataset(args, tokenizer, mode=mode, **vqa_kwargs)
        ]
        self.dataset = ConcatDataset(self.ds_list)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]