import re
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import SimpleITK as sitk
from skimage.transform import resize
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
import os
join = os.path.join
import multiprocessing as mp


def crop_or_pad_3D(image, mask, target_shape=(32, 256, 256)):
    """Crops or pads a 3D image based on a 3D mask.

    Args:
        image (numpy array): 3D image to be cropped or padded.
        mask (numpy array): 3D mask with the same shape as the image.
        target_shape (tuple): Target shape of the output image and mask.

    Returns:
        tuple: A tuple containing the cropped or padded image and mask.
    """

    # Find non-zero indices in the mask
    non_zero_indices = np.where(mask != 0)

    # Calculate the bounding box of the non-zero region
    min_z = np.min(non_zero_indices[0])
    max_z = np.max(non_zero_indices[0])
    min_y = np.min(non_zero_indices[1])
    max_y = np.max(non_zero_indices[1])
    min_x = np.min(non_zero_indices[2])
    max_x = np.max(non_zero_indices[2])

    # Calculate the size of the non-zero region
    non_zero_size = (max_z - min_z + 1, max_y - min_y + 1, max_x - min_x + 1)
    # print(f'{non_zero_size=}')
    # crop the roi
    image = image[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]
    mask = mask[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]

    # crop or pad the first dimension
    if non_zero_size[0] < target_shape[0]:
        # pad the first dimension
        pad_z = (target_shape[0] - non_zero_size[0]) // 2
        pad_z = (pad_z, target_shape[0] - non_zero_size[0] - pad_z)
        image = np.pad(image, ((pad_z[0], pad_z[1]), (0, 0), (0, 0)), mode='constant')
        mask = np.pad(mask, ((pad_z[0], pad_z[1]), (0, 0), (0, 0)), mode='constant')
        assert image.shape[0] == target_shape[0], mask.shape[0] == target_shape[0]
        # print(f'pad the first dimension: {image.shape=}, {mask.shape=}')
    else:
        # center crop the first dimension
        crop_z = (non_zero_size[0] - target_shape[0]) // 2
        # don't use min_z here, because the image has been cropped
        image = image[crop_z:crop_z + target_shape[0], :, :]
        mask = mask[crop_z:crop_z + target_shape[0], :, :]
        assert image.shape[0] == target_shape[0], mask.shape[0] == target_shape[0]
        # print(f'crop the first dimension: {image.shape=}, {mask.shape=}')
    
    # crop or pad the second dimension
    if non_zero_size[1] < target_shape[1]:
        # pad the second dimension
        pad_y = (target_shape[1] - non_zero_size[1]) // 2
        pad_y = (pad_y, target_shape[1] - non_zero_size[1] - pad_y)
        image = np.pad(image, ((0, 0), (pad_y[0], pad_y[1]), (0, 0)), mode='constant')
        mask = np.pad(mask, ((0, 0), (pad_y[0], pad_y[1]), (0, 0)), mode='constant')
        assert image.shape[1] == target_shape[1], mask.shape[1] == target_shape[1]
        # print(f'pad the second dimension: {image.shape=}, {mask.shape=}')
    else:
        # center crop the second dimension
        crop_y = (non_zero_size[1] - target_shape[1]) // 2
        image = image[:, crop_y:crop_y + target_shape[1], :]
        mask = mask[:, crop_y:crop_y + target_shape[1], :]
        assert image.shape[1] == target_shape[1], mask.shape[1] == target_shape[1]
        # print(f'crop the second dimension: {image.shape=}, {mask.shape=}')

    # crop or pad the third dimension
    if non_zero_size[2] < target_shape[2]:
        # pad the third dimension
        pad_x = (target_shape[2] - non_zero_size[2]) // 2
        pad_x = (pad_x, target_shape[2] - non_zero_size[2] - pad_x)
        image = np.pad(image, ((0, 0), (0, 0), (pad_x[0], pad_x[1])), mode='constant')
        mask = np.pad(mask, ((0, 0), (0, 0), (pad_x[0], pad_x[1])), mode='constant')
        assert image.shape[2] == target_shape[2], mask.shape[2] == target_shape[2]
        # print(f'pad the third dimension: {image.shape=}, {mask.shape=}')
    else:
        # center crop the third dimension
        crop_x = (non_zero_size[2] - target_shape[2]) // 2
        image = image[:, :, crop_x:crop_x + target_shape[2]]
        mask = mask[:, :, crop_x:crop_x + target_shape[2]]
        assert image.shape[2] == target_shape[2], mask.shape[2] == target_shape[2]
        # print(f'crop the third dimension: {image.shape=}, {mask.shape=}')

    return image, mask


nii_path = 'imagesTr'
seg_path = 'organ_mask/organ_mask_imagesTr'
save_path = 'imagesTr-Abd256'
os.makedirs(save_path, exist_ok=True)

img_names = sorted(os.listdir(nii_path))
img_names = [i for i in img_names if i.endswith('.nii.gz')]
img_names = [i for i in img_names if not os.path.isfile(join(save_path, i.replace('.nii.gz','.npy')))]
# organ mask available
img_names = [i for i in img_names if os.path.isfile(join(seg_path, i))]
print('Number of images:', len(img_names))
input('Press Enter to start processing images...')

# for img_name in img_names[0:2]:
def process_img(img_name):
    img_path = join(nii_path, img_name)
    img_sitk = sitk.ReadImage(img_path)
    img_data = sitk.GetArrayFromImage(img_sitk)
    # load and resize organ mask
    organ_sitk = sitk.ReadImage(join(seg_path, img_name))
    organ_data = sitk.GetArrayFromImage(organ_sitk)
    # select abdominal organs: https://github.com/wasserth/TotalSegmentator#class-details
    organ_data[~np.isin(organ_data, [1, 2, 3, 4, 5, 6, 7, 18])] = 0

    if img_data.shape == organ_data.shape:
        # intensity clip (adjust window level and width to 40 and 400) and normalization
        img_data = np.clip(img_data, -160.0, 240.0)
        img_data = (img_data - np.min(img_data))/ (np.max(img_data) - np.min(img_data))
        # print(img_name, 'img intensity range:', np.min(img_data), np.max(img_data)) # [0, 1]
        assert np.max(img_data) <= 1.0 and np.min(img_data) >= 0.0  # check the intensity range

        # uniform spacing
        spacings = img_sitk.GetSpacing() # (x, y, z)
        to_spacing = 5
        current_shape = img_data.shape # (z, y, x)
        current_spacing = spacings[-1]
        if current_spacing != to_spacing or current_shape[1:] != (512, 512):    
            # resize the spacing to 5mm and shape to (z, 512, 512)
            new_shape = (int(current_shape[0] * current_spacing / to_spacing), 512, 512)
            new_img_data = resize(img_data, new_shape, order=3, preserve_range=True, anti_aliasing=False).astype(img_data.dtype)
            new_organ_data = resize(organ_data, new_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            # print(img_name, 'original shape:', img_data.shape, 'new shape:', new_img_data.shape, 'organ mask shape:', new_organ_data.shape)
        else:
            new_img_data = img_data
            new_organ_data = organ_data
        # crop the image and mask based on the coords. If the size is smaller than (32, 256, 256), pad the image and mask
        target_roi_size = (64, 256, 256)
        img_roi, organ_roi = crop_or_pad_3D(new_img_data, new_organ_data, target_shape=target_roi_size)
        np.save(join(save_path, img_name.replace('.nii.gz','.npy')), np.expand_dims(img_roi, 0))
        case_id = int(re.findall(r'\d+', img_name)[0])
        if case_id % 10 == 0:
            # save the cropped image and mask
            sitk_img_roi = sitk.GetImageFromArray(img_roi)
            sitk_img_roi.SetOrigin(img_sitk.GetOrigin())
            sitk_img_roi.SetDirection(img_sitk.GetDirection())
            # update the image spacing
            sitk_img_roi.SetSpacing((1, 1, to_spacing))
            sitk_organs_roi = sitk.GetImageFromArray(organ_roi)
            sitk_organs_roi.CopyInformation(sitk_img_roi)
            sitk.WriteImage(sitk_img_roi, join(save_path, img_name.replace('.nii.gz', '_roi.nii.gz')))
            sitk.WriteImage(sitk_organs_roi, join(save_path, img_name.replace('.nii.gz', '_roi_seg.nii.gz')))
        
    else:
        print(img_name, 'img shape', img_data.shape, 'organ mask shape:', organ_data.shape)
        
if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)
    with mp.Pool(processes=10) as pool:
        r = list(tqdm(pool.imap(process_img, img_names), total=len(img_names)))