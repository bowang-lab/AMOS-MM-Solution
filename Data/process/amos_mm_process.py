import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
import json
from collections import OrderedDict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Global paths
image_path = '/home/jma/Documents/medvlm3d/imagesTr/'
t_path = '/home/jma/Documents/medvlm3d/imagesTrProcessed'  # Adjust window level and window size
os.makedirs(t_path, exist_ok=True)
json_path = '/home/jma/Documents/medvlm3d/AMOSMM_dataset.json'

# Load metadata
with open(json_path) as f:
    meta_info = json.load(f)
train_meta_infos = meta_info['validation']

def process_image(train_meta):
    """Process one training meta entry."""
    img_path = train_meta['image']
    file_name = os.path.basename(img_path)
    
    # Read image using SimpleITK
    img_sitk = sitk.ReadImage(img_path)
    img_data = sitk.GetArrayFromImage(img_sitk)
    
    # Process 4D images separately if needed
    if len(img_data.shape) == 4:
        print(file_name, img_data.shape)
        img_data = img_data[1]  # Example: select the second volume
    else:
        print(file_name, img_data.shape)
    
    # Apply clipping and normalization
    img_data = np.clip(img_data, -160.0, 240.0)
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
    
    # Resize image data
    image_np = resize(img_data, (32, 256, 256), anti_aliasing=True)
    image_np = np.expand_dims(image_np, 0)
    
    # Create save folder and save the processed image
    folder_name = file_name.split('.nii.gz')[0]
    save_folder = os.path.join(t_path, folder_name)
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, folder_name + '.npy'), image_np)
    
# Use ProcessPoolExecutor for parallel processing
with ProcessPoolExecutor(max_workers=16) as executor:
    # Wrap the executor.map with tqdm for progress bar
    list(tqdm(executor.map(process_image, train_meta_infos), total=len(train_meta_infos)))
