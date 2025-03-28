import json
import os
import re
from openai import OpenAI
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def is_non_empty_json(file_path):
    """
    Checks if the given JSON file exists and is non-empty.
    Returns True if the file exists, is a valid JSON, and contains data.
    Otherwise, returns False.
    """
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return bool(data)  # Returns False if the JSON is empty
    except (json.JSONDecodeError, IOError):
        return False

def clean_json_markdown(output):
    """
    Removes surrounding code fence markers (```json ... ```).
    """
    output = output.strip()
    if output.startswith("```json"):
        output = output[7:]
    if output.endswith("```"):
        output = output[:-3]
    if output.endswith("```json"):
        output = output[:-7]
    cleaned_output = output.strip()
    return cleaned_output

def clean_json_string(json_string):
    """
    Removes/escapes certain characters that may break JSON parsing.
    """
    cleaned_string = re.sub(r'\t', 'x', json_string)
    cleaned_string = re.sub(r'[\u00d7\u2715]', 'x', cleaned_string)
    cleaned_string = re.sub(r'[\x00-\x1f\x7f]', '', cleaned_string)
    return cleaned_string

# OpenAI API client (do not modify)
client = OpenAI(api_key="")

MODEL_NAME = "gpt-4o"
SYSTEM_PROMPT = "You are processing radiology CT scan reports and need to extract structured information from findings."
USER_PROMPT = """
Convert each finding into a triplet with the following structure:

{Entity, Position, Exist}

- Entity: The medical finding or abnormality (e.g., "nodule," "effusion," "atelectasis," "fracture," "mass," "consolidation," "fluid," "lesions," "cyst," "calcifications"). Use null if no specific entity is described.
- Position: The anatomical location (e.g., "right lower lobe," "liver," "mediastinum," "pleura," "pericardium," "right lobe of liver," "left lobe," "tail of the pancreas," "pericapsular area of the spleen"). Use null if the position is not specified.
  - **Note:** If the report mentions specific segments (e.g., "segment 6 of the right lobe of the liver"), reduce the specificity and use broader anatomical terms (e.g., "right lobe of liver," "left lobe," "right lung," "left lung," "lower right lobe"). Do not include segment numbers.
- Exist: Whether the finding is present (true) or explicitly absent (false).

Guidelines:
- If the finding states "no," "negative for," or "normal," it should be marked as negative.
- If a condition is explicitly described (e.g., "A 2 cm nodule in the left lung"), it is positive.
- Use null for missing information in Entity or Position.

Examples:
- "2 cm nodule in the right lower lobe" → {"nodule", "right lower lobe", true}
- "No pleural effusion or pneumothorax" → {"pleural effusion", null, false}, {"pneumothorax", null, false}
- "Lungs are clear" → {null, "lungs", false}
- "Multiple liver lesions noted" → {"lesions", "liver", true}
- "No focal mass in the pancreas" → {"mass", "pancreas", false}
- "A nodular low-density lesion is observed in segment 6 of the right lobe of the liver" → {"low-density lesion", "liver", true}
- "A low-density cystic mass in the tail of the pancreas" → {"cystic mass", "pancreas", true}

Your output should be in JSON format with the following structure:
{
  "Finding": Triplet
}

For example, the finding:
"2 cm nodule in the right lower lobe"
becomes
{
  "2 cm nodule in the right lower lobe": ["nodule", "right lower lobe", true]
}

If a finding canno't reasonably be made into a single triplet, then you can split the finding. For example:
"bladder with no thickening or nodules"
becomes
{
  "bladder with no thickening": ["thickening", "bladder", false],
  "bladder with no nodules": ["nodules", "bladder", false]
}

Now, extract triplets for the following CT findings and return them in JSON format:
"""

def extract_triplets_with_gpt(report_text):
    """
    This function sends the report_text to GPT,
    which returns a JSON of extracted triplets.
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{USER_PROMPT}\n\n{report_text}"}
            ],
        )
        return completion.choices[0].message.content.strip()
    except TypeError as e:
        print(f"TypeError: {e}. Ensure the response is properly formatted.")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

def process_json():
    """
    1. Read the main JSON file that contains a 'train' key.
    2. For each entry in 'train', read the text JSON file (pointed to by 'text').
    3. For each section (e.g., 'abdomen', 'pelvis') in the text JSON, prompt GPT to extract triplets.
    4. Save the GPT output to a new JSON file with the same base path, but add '_triplet' before .json.
    5. If the new JSON file already exists, skip processing.
    """
    main_json_path = "/fs01/home/junma/MedicalVLM/Data/AMOSMM_corr.json"
    
    # Load the main JSON
    with open(main_json_path, "r") as f:
        data = json.load(f)

    # Process only the "train" portion
    train_data = data.get("train", [])
    for entry in tqdm(train_data):
        text_path = entry.get("text", None)
        if not text_path:
            continue

        # Construct new path for triplet JSON
        if text_path.endswith(".json"):
            new_path = text_path.replace(".json", "_triplet.json")
        else:
            new_path = text_path + "_triplet"

        # Check if the triplet file already exists to avoid reprocessing
        if is_non_empty_json(new_path):
            print(f"{new_path} already exists and is non-empty. Skipping.")
            continue

        # Read the text JSON (e.g., {"abdomen": "...", "pelvis": "..."})
        with open(text_path, "r") as tf:
            text_sections = json.load(tf)

        # Prepare a dictionary to hold the GPT outputs
        triplet_dict = {}

        # For each key in the text sections, run GPT
        for section_name, section_text in text_sections.items():
            if section_text.strip():
                try:
                    gpt_output = extract_triplets_with_gpt(section_text)
                    # Find the first curly brace to avoid preceding text
                    i = gpt_output.find("{")
                    # Clean up the JSON string
                    gpt_output = clean_json_string(clean_json_markdown(gpt_output[i:]))
                    # Convert to Python dict
                    triplet_dict[section_name] = json.loads(gpt_output)
                except Exception as e:
                    print(f"Error processing {text_path}, section '{section_name}': {e}")
                    triplet_dict[section_name] = {}
            else:
                triplet_dict[section_name] = ""

        # Save output to the new JSON file
        with open(new_path, "w") as out_f:
            json.dump(triplet_dict, out_f, indent=2)

        print(f"Triplet JSON saved to {new_path}")

if __name__ == "__main__":
    process_json()
