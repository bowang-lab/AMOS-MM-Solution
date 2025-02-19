import numpy as np
import SimpleITK as sitk
from .triplets import COMMON_TRIPLET_STANDARD

REGION_TO_ORGAN_IDS_MAPPING = {
    "abdomen": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, # Spleen, Kidneys, Gallbladder, Liver, Stomach, Pancreas, Adrenal glands
        18, 19, 20, # Small bowel, Duodenum, Colon
        23, 24, # Kidney cysts left and right
        52, 63, 64 # Aorta, Inferior vena cava, Portal vein and splenic vein
    ],
    "pelvis": [
        21, 22, # Urinary bladder, Prostate
        25, 26, # Sacrum, Vertebrae S1
        65, 66, 67, 68, # Iliac arteries and veins
        75, 76,
        77, 78, # Hip left, Hip right
        79, # Spinal cord (if including parts of spinal cord in the pelvis)
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89 # Gluteus and iliopsoas muscles
    ],
    "chest": [
        10, 11, 12, 13, 14, # Lungs
        15, 16, 17, # Esophagus, Trachea, Thyroid gland
        51, 52, 53, # Heart, Aorta, Pulmonary vein
        54, 55, 56, 57, 58, 59, 60, # Brachiocephalic trunk, Subclavian arteries, Common carotid arteries, Brachiocephalic veins
        61, 62, 63, 64, # Atrial appendage left, Superior vena cava, Inferior vena cava, Portal vein and splenic vein
        92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, # Ribs
        116, 117 # Sternum, Costal cartilages
    ]
}

def read_numpy_or_dicom(path, ext="npy"):
    if ext == "npy":
        img = np.load(path)  # nomalized 0-1, C,D,H,W
    else:
        img_sitk = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img_sitk)
        del img_sitk
    return img

def make_triplet_prompt(entity, position):
    if entity is None and position is not None:
        prompt = f"Is the {position} normal?"
    elif entity is not None and position is None:
        prompt = f"Can you observe {entity} in this CT scan?"
    else:
        prompt = f"Is there {entity} in the {position}?"
    return prompt

def triplet_prompt(data, organ=None, standardize=False):
    prompts = []
    labels = []
    
    if organ in COMMON_TRIPLET_STANDARD.keys():
        indices_found = [False for _ in range(len(list(COMMON_TRIPLET_STANDARD[organ].keys())))]
        labels_for_indices = [False for _ in range(len(indices_found))]

    for _, values in data.items():
        entity, position, exist = values
        
        found_for_this_example = False

        if standardize and organ in COMMON_TRIPLET_STANDARD.keys():
            common_triplets = COMMON_TRIPLET_STANDARD[organ]
            for j, (triplet, other_names) in enumerate(common_triplets.items()):
                if [entity,position] in other_names:
                    indices_found[j] = True
                    labels_for_indices[j] = exist
                    found_for_this_example = True
                    break

        if not found_for_this_example:
            prompt = make_triplet_prompt(entity, position)
            prompts.append(prompt)
            labels.append(str(exist))

    if standardize and organ in COMMON_TRIPLET_STANDARD.keys():
        for j, found in enumerate(indices_found):
            triplets = list(common_triplets.keys())
            triplet = triplets[j]
            prompts.insert(0, make_triplet_prompt(triplet[0], triplet[1]))
            if found:
                labels.insert(0, str(labels_for_indices[j]))
            else:
                labels.insert(0, "False")

    # add numbering
    prompts = [f"{i}. {p}" for i, p in enumerate(prompts)]
    labels = [f"{i}. {l}" for i, l in enumerate(labels)]

    return "\n".join(prompts), "\n".join(labels)