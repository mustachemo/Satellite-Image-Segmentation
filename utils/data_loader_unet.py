import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import rgb_to_grayscale
from tqdm import tqdm

def load_and_process_files(image_dir, mask_dir, prefix='train'):
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
            print(f'image: {file_name}.png mask: {file_name}_mask.png not found, continuing...')
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
    
    # Save the images and masks
    np.save(f'prepped_data/{prefix}images.npy', np.array(images))
    np.save(f'prepped_data/{prefix}masks.npy', np.array(masks))