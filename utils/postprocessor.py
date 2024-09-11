from LaMed.src.dataset.multi_dataset import read_image

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


def all_items_in_string(lst, string):
    string_lower = string.lower()
    return all(item.lower() in string_lower for item in lst)

class PostProcessor():
    def __init__(self, results, post_process_list, model=None, tokenizer=None, dataset=None, organs=["abdomen", "chest", "pelvis"]):
        self.results = results
        self.post_process_list = post_process_list
        self.organs = organs
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        for organ in organs:
            if f"no_postprocessing-{organ}" not in results.keys():
                results[f"no_postprocessing-{organ}"] = results[f"generated-{organ}"]
            
            if "green_" + organ in self.results.keys():

                self.results["green_" + organ] = [-1] * len(self.results["green_" + organ])
                self.results["explanation_" + organ] = [""] * len(self.results["explanation_" + organ])

    def focused_inference(self):
        data_list = [i["image"] for i in self.dataset.data_list]
        names = self.results["names"]
        chest = self.results["generated-chest"]

        import monai.transforms as mtf
        import torch
        import os
        transform = mtf.Compose(
        [
            mtf.ToTensor(dtype=torch.bfloat16),
            mtf.Resize((32, 256, 256))
        ])

        qa = [
        {
            "q":"Do you observe fluid in the chest cavities? What about in the thoracic, pleural cavity, OR pericardium? Please answer with 'yes' if you suspect you observe ANY of them, even if you are not very confident.", 
            "a": "Fluid is observed in both chest cavities, thoracic, right and left pleural cavity, or pericardium,  consistent with pleural effusion. "
        },
        {
            "q": "Do you observe patchy linear OR high-density shadows in the lungs? Please answer with 'yes' if you suspect you observe ANY of them, even if you are not very confident.", 
            "a": "Patchy linear and high-density opacities and shadows are observed in the lungs, potentially indicating conditions such as pneumonia, pulmonary edema, or interstitial lung disease. "
        }
        ]
        from tqdm import tqdm
        for i, name in tqdm(enumerate(names)):
            image_path = data_list[i]
            if image_path.startswith('/'):
                image_abs_path = image_path 
            else:
                image_abs_path = os.path.join(self.data_root, image_path)
            image = read_image(image_abs_path)
            image = transform(image).unsqueeze(0).to(self.model.device, dtype=torch.bfloat16)

            for q_ in qa:
                conversation = [{"role": "system", "content": "You are an AI assistant trained to act as a thoracic radiologist. Only output 'yes' or 'no' answers, nothing else"},
                    {"role": "user", "content": q_["q"]}]
                input_ids = self.tokenizer.apply_chat_template(conversation, tokenize=False)
                input_ids = "<im_patch>" * 256 + input_ids
                input_ids = self.tokenizer(
                    input_ids, return_tensors="pt"
                )["input_ids"].to(self.model.device)   

                generation = self.model.generate(image, input_ids, segs=None, max_new_tokens=5, do_sample=True, top_p=0.9, temperature=0.5)
                pred = self.tokenizer.batch_decode(generation, skip_special_tokens=True)[0].strip().lower()

                if ("yes" in pred) and isinstance(chest[i], str) and (q_["a"] not in chest[i]):
                    chest[i] = q_["a"] + chest[i] 
                    
        self.results["generated-chest"] = chest 


    def run(self):
        if "simple_normal" in self.post_process_list:
            
            abdomen = self.results["generated-abdomen"]
            for i, example in enumerate(abdomen): 
                for organ in ["Liver", "Spleen", "Gallbladder", "Bile duct", "Pancreas", "Kidneys"]:
                    if organ.casefold() not in example.casefold():
                        abdomen[i] += f" The {organ} is normal."
            self.results["generated-abdomen"] = abdomen

        elif "complex_normal" in self.post_process_list:
            print("Doing complex normal")
            self.complex_normal()

        if "focused_inference" in self.post_process_list:
            print("Doing focused inference")
            self.focused_inference()

        return self.results
    
    def complex_normal(self):
        chest = self.results["generated-chest"]
        for i, example in enumerate(chest):
            if isinstance(example, float): # check nans
                continue  
            example = example.split(".")
            for normality in CHEST_MAPPING.keys():
                is_in = False
                for finding in example:
                    if all_items_in_string(normality, finding):
                        is_in = True
                        break
                if is_in == False:
                    example.insert(0, " " + CHEST_MAPPING[normality])
            chest[i] = ".".join(example).strip()
        self.results["generated-chest"] = chest

        pelvis = self.results["generated-pelvis"]
        for i, example in enumerate(pelvis):
            if isinstance(example, float): # check nans
                continue  
            example = example.split(".")
            for normality in PELVIS_MAPPING.keys():
                is_in = False
                for finding in example:
                    if all_items_in_string(normality, finding):
                        is_in = True
                        break
                if is_in == False:
                    example.insert(0, " " + PELVIS_MAPPING[normality])
            pelvis[i] = ".".join(example).strip()
        self.results["generated-pelvis"] = pelvis

        abdomen = self.results["generated-abdomen"]
        for i, example in enumerate(abdomen):
            if isinstance(example, float): # check nans
                continue  
            example = example.split(".")
            for normality in ABDOMEN_MAPPING.keys():
                is_in = False
                for finding in example:
                    if all_items_in_string(normality, finding):
                        is_in = True
                        break
                if is_in == False:
                    example.insert(0, " " + ABDOMEN_MAPPING[normality])
            abdomen[i] = ".".join(example).strip()
        self.results["generated-abdomen"] = abdomen