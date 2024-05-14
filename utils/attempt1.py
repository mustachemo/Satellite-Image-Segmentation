# prepare the images and their masks for tensorlfow U-Net model
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image_and_mask(image_path, mask_path, target_size=(256, 256)):
    """Load and resize an image and its corresponding mask."""
    # Load image and mask
    image = load_img(image_path, target_size=target_size, color_mode="rgb")
    mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
    
    # Convert to numpy array
    image = img_to_array(image)
    mask = img_to_array(mask)
    
    # Normalize image
    image = image / 255.0
    # Normalize mask and convert to binary format
    mask = mask / 255.0
    mask = (mask > 0.5).astype(int)
    
    return image, mask

def create_dataset(image_dir, mask_dir, batch_size=8, target_size=(256, 256)):
    """Create a TensorFlow dataset from images and masks directories."""
    # Get all image and mask paths
    image_paths = [os.path.join(image_dir, fname) for fname in sorted(os.listdir(image_dir))]
    mask_paths = [os.path.join(mask_dir, fname) for fname in sorted(os.listdir(mask_dir))]
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    # Load and preprocess images
    def process_path(image_path, mask_path):
        return load_image_and_mask(image_path, mask_path, target_size)
    
    dataset = dataset.map(lambda image_path, mask_path: tf.numpy_function(
        process_path, [image_path, mask_path], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

# Usage

