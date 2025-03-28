from LaMed.src.dataset.multi_dataset import read_image
from LaMed.src.dataset.utils import triplet_prompt
from LaMed.src.dataset.prompts import prompt_templates

from monai.transforms import Compose, ToTensor, Resize
import torch
from tqdm import tqdm

COMMON_TRIPLETS = {
    'chest': {
        ("Nodules are seen in the lungs.", "No nodules are seen in the lungs."): ['nodules', 'lungs', None],
        ("Fluid is seen in the pleural cavities or bilateral pleural cavities.", "No fluid is seen in the pleural cavities."): ['fluid', 'pleural cavities', None],
        ("Linear or patchy shadows are seen in the lung (any of the lobes).", "No shadows are seen in the lung."): ['shadows', 'lungs', None],
        ("Pleural effusion is observed in the heart, chest, or pleura.", "No pleural effusion is seen."): ['pleural effusion', None, None]
    },
    'abdomen': {
        ("Dilation of the bile ducts is observed.", "No dilation of the bile ducts is observed."): ['dilation', 'bile ducts', None],
        ("Dilation of the intestines is observed.", "No dilation of the intestines is observed."): ['dilation', 'intestines', None],
        ("Dilation of the renal pelvis or calyces is observed.", "No dilation of the renal pelvis or calyces is observed."): ['dilation', 'renal pelvis or calyces', None],
        ("Dilation of the ureter is observed.", "No dilation of the ureter is observed."): ['dilation', 'ureter', None],
        ("Enlargement of the spleen is observed.", "No enlargement of the spleen is observed."): ['enlargement', 'spleen', None],
        ("Enlargement of the gallbladder is observed.", "No enlargement of the gallbladder is observed."): ['enlargement', 'gallbladder', None],
        ("Enlargement of lymph nodes in the abdomen is observed.", "No enlargement of lymph nodes in the abdomen is observed."): ['enlargement', 'lymph nodes in abdomen', None],
        ("Enlargement of the kidneys is observed.", "No enlargement of the kidneys is observed."): ['enlargement', 'kidneys', None],
        ("Enlargement of the pancreas is observed.", "No enlargement of the pancreas is observed."): ['enlargement', 'pancreas', None],
        ("Density lesions (either high or low) in the liver are observed.", "No density lesions in the liver are observed."): ['lesions', 'liver', None],
        ("Density lesions (either high or low) in the kidney are observed.", "No density lesions in the kidney are observed."): ['lesions', 'kidney', None],
        ("Density lesions (either high or low) in the gallbladder are observed.", "No density lesions in the gallbladder are observed."): ['lesions', 'gallbladder', None],
        ("Density lesions (either high or low) in the ureter are observed.", "No density lesions in the ureter are observed."): ['lesions', 'ureter', None]
    }, "pelvis": {
        ("Bladder wall thickening is observed.", "No bladder wall thickening is observed."): ['thickening', 'bladder wall', None],
        ("Fluid accumulation is observed in the pelvic cavity.", "No fluid accumulation is observed in the pelvic cavity."): ['fluid accumulation', 'pelvic cavity', None],
        ("Prostate enlargement is observed.", "No prostate enlargement is observed."): ['enlargement', 'prostate', None],
        ("Enlarged lymph nodes are observed in the pelvic cavity.", "No enlarged lymph nodes are observed in the pelvic cavity."): ['enlarged lymph nodes', 'pelvic cavity', None],
        ("Calcifications are observed in the prostate.", "No calcifications are observed in the prostate."): ['calcifications', 'prostate', None]
    }
}

CHEST_MAPPING = {
    ("lung fields",): "The Lung fields are clear and normal with no evidence of consolidation",
    ("heart",): "The heart size and shape is normal and within limits. The heart is normal",
    ("mediastinum",): "No mediastinal widening",
    ("lymph nodes", "mediastinum"): "No enlarged lymph nodes in the mediastinum or the bilateral hilum",
    ("bilateral pulmonary hila",): "No enlargement is observed in the bilateral pulmonary hila",
    ("trachea",): "The trachea is unobstructed and smooth",
    ("bronchi",): "The main bronchi are unobstructed and smooth, with no signs of stenosis",
    ("airways",): "The airways are unobstructed",
    ("lung parenchyma",): "No infiltrative or space-occupying lesions are seen in the lung parenchyma",
    ("pleural cavities", "pleural effusion"): "No pleural effusion is seen in both pleural cavities or bilateral pleural cavities",
    ("chest bones",): "Chest bones are normal",
    ("chest", "symmetrical"): "The chest is symmetrical",
    ("seminal vesicles",): "No obvious abnormalities in the seminal vesicles"
}

PELVIS_MAPPING = {
    ("bladder"): "No obvious abnormalities are seen in the bladder",
    ("retroperitoneum", "lymph"): "No enlarged lymph nodes are seen in the retroperitoneum",
    ("fat gap",): "The surrounding fat gap is clear",
    ("pelvic region",): "The pelvic region is normal, with no soft tissue mass",
    ("bilateral seminal vesicles",): "The bilateral seminal vesicles are symmetrical, with no abnormal density inside",
    ("bladder", "fill"): "Bladder is filled",
    ("bladder wall"): "The bladder wall is smooth",
    ("prostate",): "The prostate position, volume, and size are within the normal range, with a smooth contour and uniform density",
    ("bladder-vesical junction",): "The bladder-vesical junction is clear",
    ("intestine",): "The intestine is normal",
    ("pelvic effusion"): "No obvious pelvic effusion is seen",
    ("pelvic cavity", "lymph"): "No lymph nodes are seen in the pelvic cavity",
    ("uterus",): "The uterus is normal, with normal density and no abnormalities",
    ("bladder", "nodules"): "No nodules are observed in the bladder",
    ("seminal vesicles",): "The angle between the bladder and seminal vesicles is clear",
    ("bilateral adnexal",): "No abnormalities are seen in the bilateral adnexal regions",
    ("bladder trigone",): "The bladder trigone is clear",
}

ABDOMEN_MAPPING = {
    ("left kidney",): "The left kidney is normal, with no abnormal density in the parenchyma",
    ("right kidney",): "The right kidney is normal, with no abnormal density in the parenchyma",
    ("renal pelvis", "calyces"): "The renal pelvis and calyces are not dilated",
    ("retroperitoneum", "lymph"): "No enlargement of lymph nodes in the retroperitoneum",
    ("pancreas",): "The size and shape of the pancreas are normal, with uniform density",
    ("liver",): "The liver is normal, with a smooth surface",
    ("gallbladder",): "The gallbladder is normal, with a normal size and shape",
    ("common bile duct"): "The common bile duct and intrahepatic bile duct are normal, and not dilated",
    ("spleen",): "The spleen is of normal shape, size, and density",
    ("intestinal wall",): "No obvious thickening of the intestinal wall",
    ("perirenal fat gap",): "The perirenal fat gap is clear",
    ("bilateral ureteral"): "No abnormal density foci are observed in the bilateral ureteral course",
    ("pancreatic duct",): "The main pancreatic duct is not dilated",
    ("abdominal cavity", "lymph"): "No enlarged lymph nodes are seen in the abdominal cavity",
    ("intestines",): "No abnormal morphology of the intestines is observed",
}

organ_mappings = {
    "generated-chest": CHEST_MAPPING,
    "generated-pelvis": PELVIS_MAPPING,
    "generated-abdomen": ABDOMEN_MAPPING,
}

def string_to_list(string):
    bool_list = []
    for line in string.strip().splitlines():
        parts = line.split()
        value_str = parts[-1]
        bool_list.append(value_str == "True")
    return bool_list

def all_items_in_string(lst, string):
    string_lower = string.lower()
    return all(item.lower() in string_lower for item in lst if isinstance(item, str))

class PostProcessor():
    def __init__(self, results, post_process_list, dataset=None, organs=["abdomen", "chest", "pelvis"]):
        self.results = results
        self.post_process_list = post_process_list
        self.organs = organs
        self.dataset = dataset
        for organ in organs:
            if f"no_postprocessing-{organ}" not in results.keys():
                results[f"no_postprocessing-{organ}"] = results[f"generated-{organ}"]
            
            if "green_" + organ in self.results.keys():

                self.results["green_" + organ] = [-1] * len(self.results["green_" + organ])
                self.results["explanation_" + organ] = [""] * len(self.results["explanation_" + organ])

        if "focused_inference" in post_process_list:
            # get triplet model and tokenizer
            import torch
            from LaMed.src.model.language_model import LamedPhi3ForCausalLM
            from transformers import AutoTokenizer
            triplet_model_path = "/checkpoint/datasets.damaged/med-img-data/amosmm/trained/paper/triplet_model_4"
            self.model = LamedPhi3ForCausalLM.from_pretrained(
                triplet_model_path,
                cache_dir='/scratch/ssd004/datasets/med-img-data/amosmm/trained/cache/',
                torch_dtype=torch.bfloat16,
                device_map='auto',
                trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                triplet_model_path,
                cache_dir='/scratch/ssd004/datasets/med-img-data/amosmm/trained/cache/',
                model_max_length=128,
                padding_side="right",
                use_fast=False,
                trust_remote_code=True
            )

    def focused_inference(self):
        # Prepare image paths and names
        data_list = [item["image"] for item in self.dataset.data_list]
        names = self.results["names"]

        transform = Compose([
            ToTensor(dtype=torch.bfloat16),
            Resize((32, 256, 256))
        ])

        def construct_prompt(organ, common_triplets):
            """Construct the prompt based on the triplet template and common triplets."""
            prompt_template = prompt_templates['triplet'][organ]
            sentences = prompt_template.split('.')
            system = sentences[0] + '.'
            content = '.'.join(sentences[1:])
            prompt_addition, _ = triplet_prompt(common_triplets)
            content += '\n' + prompt_addition
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": content}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Loop through images and corresponding names
        for i, (name, image_path) in enumerate(tqdm(zip(names, data_list), total=len(names))):
            image = read_image(image_path)
            image = transform(image).unsqueeze(0).to(self.model.device, dtype=torch.bfloat16)

            # Process each organ's report if it exists
            for organ in self.organs:
                report_key = f"generated-{organ}"
                report_list = self.results.get(report_key)
                if report_list is None:
                    continue

                report_text = report_list[i]
                if not isinstance(report_text, str) or organ not in COMMON_TRIPLETS:
                    continue

                common_triplets = COMMON_TRIPLETS[organ]
                prompt = construct_prompt(organ, common_triplets)

                # Create input tokens by prepending image placeholders
                input_str = "<im_patch>" * 256 + prompt
                input_ids = self.tokenizer(input_str, return_tensors="pt")["input_ids"].to(self.model.device)

                # Generate output and decode
                generation = self.model.generate(
                    image, input_ids, segs=None, max_new_tokens=128,
                    do_sample=False, top_p=0.9, temperature=1
                )
                generation = self.tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
                answers_list = string_to_list(generation)

                common_triplets_list = list(common_triplets.values())
                common_findings_list = list(common_triplets.keys())

                # Update the report based on the answers
                for answer, (finding_key, triplet_value) in zip(answers_list, zip(common_findings_list, common_triplets_list)):
                    if any(all_items_in_string(triplet_value, part) for part in report_text.split('.')):
                        continue
                    # Append the finding based on the answer (True selects the first item)
                    finding_to_append = finding_key[0] if answer else finding_key[1]
                    report_text += f" {finding_to_append}"

                report_list[i] = report_text
                self.results[report_key] = report_list
    
    def normality(self):
        def add_missing_findings(text: str, mapping: dict) -> str:
            parts = text.split(".")
            for normality, description in mapping.items():
                if not any(all_items_in_string(normality, part) for part in parts):
                    parts.insert(0, " " + description)
            return ".".join(parts).strip()
        
        for organ, mapping in organ_mappings.items():
            self.results[organ] = [
                example if isinstance(example, float) 
                else add_missing_findings(example, mapping)
                for example in self.results[organ]
            ]

    def run(self):
        if "focused_inference" in self.post_process_list:
            print("Doing focused inference")
            self.focused_inference()

        if "normality" in self.post_process_list:
            print("Doing normality")
            self.normality()

        return self.results