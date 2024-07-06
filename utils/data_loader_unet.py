import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

X_DIMENSION = 256
Y_DIMENSION = 256

def visualize_sample(image, mask):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    plt.show()
    print('Visualizing sample...')

def process_image(image_path, mask_path, image_color_mode='rgb', mask_color_mode='grayscale', make_mask_binary=True):
    try:
        image = load_img(image_path, color_mode=image_color_mode, target_size=(X_DIMENSION, Y_DIMENSION))
        mask = load_img(mask_path, color_mode=mask_color_mode, target_size=(X_DIMENSION, Y_DIMENSION))
        
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        mask = tf.convert_to_tensor(mask, dtype=tf.uint8)
        mask = tf.expand_dims(mask, axis=-1)
        if make_mask_binary:
            mask = tf.cast(mask > 0, tf.uint8)
            
        return image, mask
    except Exception as e:
        print(f'Image or mask not found: {image_path}, {mask_path}')
        return None, None


def load_and_process_files(image_dir, mask_dir, prefix='train'):
    images = []
    masks = []
    image_paths = list(image_dir.glob('*.png'))
    
    for image_path in tqdm(image_paths, total=len(image_paths), desc=f'Loading and processing {prefix} data'):
        mask_path = mask_dir / f'{image_path.stem}_mask.png'
        image, mask = process_image(image_path, mask_path)
        if image is not None and mask is not None:
            images.append(image)
            masks.append(mask)


    return images, masks

def serialize_example(image, mask):
    image_value = tf.io.encode_png(image).numpy()
    mask_value = tf.io.encode_png(mask).numpy()
    
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_value])),
        'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_value])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def write_tfrecord(filename, images, masks):
    with tf.io.TFRecordWriter(filename) as writer:
        for image, mask in zip(images, masks):
            example = serialize_example(image, mask)
            writer.write(example)

def read_tfrecord(serialized_example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_jpeg(example['image'])
    mask = tf.io.decode_jpeg(example['mask'])
    image = tf.cast(image, tf.float16) / 255.0
    mask = tf.cast(mask, tf.float16)
    return image, mask

def create_tf_dataset_from_tfrecord(tfrecord_files, batch_size=1):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = raw_dataset.map(read_tfrecord)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    image_dir = Path('data_sample/images/train')
    mask_dir = Path('data_sample/masks/train')

    images, masks = load_and_process_files(image_dir, mask_dir, prefix='train')
    write_tfrecord('prepped_data/train.tfrecord', images, masks)
    dataset = create_tf_dataset_from_tfrecord(['prepped_data/train.tfrecord'], batch_size=1)
