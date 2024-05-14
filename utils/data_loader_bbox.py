from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array

from pathlib import Path
from PIL import Image
import numpy as np
import json
from tqdm import tqdm


image_dir = Path(r'../data/images/train')
annotation_file = Path(r'../data/all_bbox.txt')

print(f"Image directory: {image_dir}")
image_paths = list(image_dir.glob('*.png'))
print(f"Number of images found: {len(image_paths)}")




# Load your training images
train_images = []
image_indices = []
for image_path in tqdm(image_paths):  # Adjust the file extension as needed
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image_indices.append(image_path.stem) # Extract the image index
    train_images.append(image)
train_images = np.array(train_images, dtype="object")

# Load your training bounding box annotations
with open(annotation_file, 'r') as f:
    all_bboxes = json.load(f)

# Match bounding boxes with images
train_annotations = []
for idx in tqdm(image_indices):
    if idx in all_bboxes:
        bboxes = all_bboxes[idx]
        train_annotations.append(bboxes)
    else:
        train_annotations.append([])  # Handle missing annotations

# Normalize bounding boxes
def normalize_bboxes(bboxes, img_width, img_height):
    normalized_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        xmin /= img_width
        xmax /= img_width
        ymin /= img_height
        ymax /= img_height
        normalized_bboxes.append([xmin, ymin, xmax, ymax])
    return normalized_bboxes

# Get the dimensions of the first image
img_width, img_height = train_images[0].shape[1], train_images[0].shape[0]

# Normalize the bounding boxes for all images
train_annotations = [normalize_bboxes(bboxes, img_width, img_height) for bboxes in train_annotations]
train_annotations = np.array(train_annotations, dtype=object)

# Check shapes of the training data
print("Shape of train_images:", train_images.shape)  # (num_images, height, width, channels)
print("Number of annotations:", len(train_annotations))  # Should match the number of images
