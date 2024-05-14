import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import rgb_to_grayscale
from tqdm import tqdm

# Directory paths
train_images_dir = './data/images/train'
train_masks_dir = './data/mask/train'
test_images_dir = './data/images/val'
test_masks_dir = './data/mask/val'

# Sort files
train_image_files = sorted(os.listdir(train_images_dir))
train_mask_files = sorted(os.listdir(train_masks_dir))
test_image_files = sorted(os.listdir(test_images_dir))
test_mask_files = sorted(os.listdir(test_masks_dir))

def load_and_process_files(image_dir, mask_dir, image_files, mask_files):
    images = []
    masks = []
    for i in tqdm(range(3117), desc='Prepping images'):
        # File correspondence check
        file_name = f'img_resize_{i}'

        # Load images
        image_path = os.path.join(image_dir, f'{file_name}.png')
        mask_path = os.path.join(mask_dir, f'{file_name}_mask.png')

        try:
            image = load_img(image_path, color_mode='rgb', target_size=(256, 256))
            mask = load_img(mask_path, color_mode='grayscale', target_size=(256, 256))
        except Exception as e:
            print(f'image: {image_path} mask: {mask_path} not found, continuing...')
            continue


        # Convert to array and normalize
        image = img_to_array(image) / 255.0
        mask = img_to_array(mask) / 255.0

        # Convert to tensors
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

        # Append to lists
        images.append(image)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Process training and testing data
train_images, train_masks = load_and_process_files(train_images_dir, train_masks_dir, train_image_files, train_mask_files)
test_images, test_masks = load_and_process_files(test_images_dir, test_masks_dir, test_image_files, test_mask_files)
# print the shapes of the images and masks
print(f'Train images shape: {train_images.shape}')
print(f'Train masks shape: {train_masks.shape}')
print(f'Test images shape: {test_images.shape}')
print(f'Test masks shape: {test_masks.shape}')