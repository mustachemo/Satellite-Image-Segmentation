import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Directory paths
images_dir = './temp_data/images/train'
masks_dir = './temp_data/mask/train'

# Get all file names
image_files = os.listdir(images_dir)
mask_files = os.listdir(masks_dir)

# Initialize lists to store images and masks
images = []
masks = []

# Load images and masks
for image_file, mask_file in zip(image_files, mask_files):
    # Load image and mask
    image = load_img(os.path.join(images_dir, image_file))
    mask = load_img(os.path.join(masks_dir, mask_file), color_mode='grayscale')  # assuming masks are grayscale

    # Convert to numpy array and normalize to 0-1
    image = img_to_array(image) / 255.0
    mask = img_to_array(mask) / 255.0

    # Append to lists
    images.append(image)
    masks.append(mask)

# Convert lists to numpy arrays
images = np.array(images)
masks = np.array(masks)

# Split into training and validation sets
images_train, images_val, masks_train, masks_val = train_test_split(images, masks, test_size=0.2, random_state=42)