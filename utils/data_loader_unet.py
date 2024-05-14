import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# Directory paths
train_images_dir = './data/images/train'
test_images_dir = './data/images/val'
train_masks_dir = './data/mask/train'
test_masks_dir = './data/mask/val'

# Get all file names
train_image_files = os.listdir(train_images_dir)
train_mask_files = os.listdir(train_masks_dir)
test_image_files = os.listdir(test_images_dir)
test_mask_files = os.listdir(test_masks_dir)


train_images = []
train_masks = []
test_images = []
test_masks = []

print('Converting images and masks to numpy arrays...')
for image_file, mask_file in tqdm(zip(train_image_files, train_mask_files)):

    image = load_img(os.path.join(train_images_dir, image_file))
    mask = load_img(os.path.join(train_masks_dir, mask_file), color_mode='grayscale')  # assuming masks are grayscale


    target_size = (256, 256)
    image = image.resize(target_size)
    mask = mask.resize(target_size)

    # Convert to numpy array and normalize to 0-1
    image = img_to_array(image) / 255.0
    mask = img_to_array(mask) / 255.0

    # Append to lists
    train_images.append(image)
    train_masks.append(mask)

print('Converting test images and masks to numpy arrays...')
for image_file, mask_file in tqdm(zip(test_image_files, test_mask_files)):

    image = load_img(os.path.join(test_images_dir, image_file))
    mask = load_img(os.path.join(test_masks_dir, mask_file), color_mode='grayscale')  # assuming masks are grayscale

    target_size = (256, 256)
    image = image.resize(target_size)
    mask = mask.resize(target_size)

    # Convert to numpy array and normalize to 0-1
    image = img_to_array(image) / 255.0
    mask = img_to_array(mask) / 255.0

    # Append to lists
    test_images.append(image)
    test_masks.append(mask)


# Convert lists to numpy arrays
train_images = np.array(train_images)
train_masks = np.array(train_masks)
test_images = np.array(test_images)
test_masks = np.array(test_masks)