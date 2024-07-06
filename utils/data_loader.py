import tensorflow as tf
from tensorflow.keras.utils import load_img # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
from configs import X_DIMENSION, Y_DIMENSION
import logging

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
        logging.warn(f'Image or mask not found: {image_path}, {mask_path}')
        return None, None


def load_and_process_files(image_dir, mask_dir, prefix='train'):
    images = []
    masks = []
    image_paths = list(image_dir.glob('*.png'))

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image_path, mask_dir / f'{image_path.stem}_mask.png'): image_path for image_path in image_paths}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Loading and processing {prefix} data'):
            image, mask = future.result()
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
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(serialize_example, image, mask) for image, mask in zip(images, masks)]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Writing TFRecord'):
                example = future.result()
                writer.write(example)

def read_tfrecord(serialized_example):
    '''Read a single example from a TFRecord file and decode it'''
    
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_png(example['image'])
    mask = tf.io.decode_png(example['mask'])
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32)
    return image, mask

def create_tf_dataset_from_tfrecord(tfrecord_files, batch_size=1):
    '''Create a TFRecord dataset from a list of TFRecord files'''
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    logging.info(f'Creted dataset from {tfrecord_files} with batch size {batch_size}...')
    logging.info('Normalized Images but not masks, as they are binary')
    dataset = raw_dataset.map(read_tfrecord)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
