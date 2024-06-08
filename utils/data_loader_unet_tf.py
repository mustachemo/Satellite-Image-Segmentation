import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from configs import X_DIMENSION, Y_DIMENSION, BATCH_SIZE

def load_and_preprocess_image(image_path, mask_path):
    image = load_img(image_path, color_mode='rgb', target_size=(X_DIMENSION, Y_DIMENSION))
    mask = load_img(mask_path, color_mode='grayscale', target_size=(X_DIMENSION, Y_DIMENSION))

    image = img_to_array(image, dtype=np.float32) / np.max(image)
    mask = img_to_array(mask, dtype=np.float32) / np.max(mask)
    mask[mask > 0.1] = 1.0

    return image, mask

def load_and_process_files(image_dir, mask_dir, batch_size=BATCH_SIZE, prefix='train'):
    image_paths = []
    mask_paths = []

    for i in range(3117):
        file_name = f'img_resize_{i}'
        image_path = os.path.join(image_dir, f'{file_name}.png')
        mask_path = os.path.join(mask_dir, f'{file_name}_mask.png')

        if os.path.exists(image_path) and os.path.exists(mask_path):
            image_paths.append(image_path)
            mask_paths.append(mask_path)
        else:
            print(f'image: {file_name}.png or mask: {file_name}_mask.png not found, continuing...')

    # Create a dataset of file paths
    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    # Define a function to load and preprocess the images and masks
    def load_and_preprocess_from_path(image_path, mask_path):
        image, mask = tf.numpy_function(load_and_preprocess_image, [image_path, mask_path], [tf.float32, tf.float32])
        image.set_shape((X_DIMENSION, Y_DIMENSION, 3))
        mask.set_shape((X_DIMENSION, Y_DIMENSION, 1))
        return image, mask

    # Apply the function to each element in the dataset
    image_mask_ds = path_ds.map(load_and_preprocess_from_path, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    dataset = image_mask_ds.batch(batch_size)
    
    # Enable prefetching
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Save the dataset as tfrecord if needed
    if prefix:
        tf.data.experimental.save(dataset, f'prepped_data/{prefix}')
    
    return dataset


# def parse_image(image_path, mask_path):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_png(image, channels=3)
#     image = tf.image.resize(image, [X_DIMENSION, Y_DIMENSION])
#     image = tf.cast(image, tf.float32) / 255.0

#     mask = tf.io.read_file(mask_path)
#     mask = tf.image.decode_png(mask, channels=1)
#     mask = tf.image.resize(mask, [X_DIMENSION, Y_DIMENSION])
#     mask = tf.cast(mask, tf.float32) / 255.0
#     mask = tf.where(mask > 0.1, 1.0, 0.0)

#     return image, mask

# def load_and_process_files(image_dir, mask_dir, prefix='train', batch_size=BATCH_SIZE, cache=True, shuffle_buffer_size=1000):
#     image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
#     mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

#     image_paths = [os.path.join(image_dir, f) for f in image_files]
#     mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

#     assert len(image_paths) == len(mask_paths), "Number of images and masks must be the same"

#     path_ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
#     dataset = path_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

#     if cache:
#         dataset = dataset.cache()

#     dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#     if prefix:
#         tf.data.experimental.save(dataset, f'prepped_data/{prefix}')

#     return dataset