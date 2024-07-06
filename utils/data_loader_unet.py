import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# from configs import X_DIMENSION, Y_DIMENSION
X_DIMENSION = 256
Y_DIMENSION = 256

def process_image(image_path, mask_path, image_color_mode='rgb', mask_color_mode='grayscale', make_mask_binary=True):
    try:
        image = load_img(image_path, color_mode=image_color_mode, target_size=(X_DIMENSION, Y_DIMENSION))
        mask = load_img(mask_path, color_mode=mask_color_mode, target_size=(X_DIMENSION, Y_DIMENSION))
    except Exception as e:
        print(f'Image or mask not found: {image_path}, {mask_path}')
        return None, None

    image = img_to_array(image, dtype=np.float32) / 255.0
    mask = img_to_array(mask, dtype=np.float32) / 255.0
    
    if make_mask_binary:
        mask = np.where(mask > 0.0, 1.0, 0.0)

    return image, mask

def load_and_process_files(image_dir, mask_dir, prefix='train'):
    images = []
    masks = []
    image_paths = list(image_dir.glob('*.png'))
    mask_paths = list(mask_dir.glob('*.png'))
    
    for image_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc=f'Loading and processing {prefix} data'):
        image, mask = process_image(image_path, mask_path)
        if image is not None and mask is not None:
            images.append(image)
            masks.append(mask)

    # with ThreadPoolExecutor() as executor:
    #     futures = list(executor.map(process_image, image_path, mask_path) for image_path, mask_path in zip(image_paths, mask_paths))

    # for image, mask in tqdm(futures):
    #     if image is not None and mask is not None:
    #         images.append(image)
    #         masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)
    
    np.save(f'prepped_data/{prefix}_images.npy', images)
    np.save(f'prepped_data/{prefix}_masks.npy', masks)

    return images, masks

def create_tf_dataset(images, masks, batch_size=32):
    def generator():
        for image, mask in zip(images, masks):
            yield image, mask

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(X_DIMENSION, Y_DIMENSION, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(X_DIMENSION, Y_DIMENSION, 1), dtype=tf.float32)
        )
    )

    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Example usage
if __name__ == "__main__":
    from pathlib import Path

    image_dir = Path('data/images/train')
    mask_dir = Path('data/masks/train')

    images, masks = load_and_process_files(image_dir, mask_dir, prefix='train')
    train_dataset = create_tf_dataset(images, masks, batch_size=32)