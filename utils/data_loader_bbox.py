from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array

from pathlib import Path
from PIL import Image
import numpy as np
import json

target_shape = (224, 224)
image_dir = Path('data/images/train')
annotation_file = Path('data/all_bbox.txt')

datagen = ImageDataGenerator(rescale=1./255)

def load_and_resize_image(image_path):
    image = Image.open(image_path).resize(target_shape)
    image = np.array(image) / 255.0  # Rescale manually
    return image

# Load your training images
train_images = []
image_indices = []
for image_path in image_dir.glob('*.png'):  # Adjust the file extension as needed
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_shape)
    image = np.array(image)
    image_indices.append(image_path.stem) # Extract the image index
    train_images.append(image)
train_images = np.array(train_images, dtype="object")

# Load your training bounding box annotations
with open(r'all_bbox.txt', 'r') as f:
    all_bboxes = json.load(f)

# Match bounding boxes with images
train_annotations = []
for idx in image_indices:
    if idx in all_bboxes:
        bboxes = all_bboxes[idx]
        train_annotations.append(bboxes)
    else:
        train_annotations.append([])  # Handle missing annotations

# Convert bounding boxes to numpy array and normalize
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

img_width, img_height = target_shape
train_annotations = [normalize_bboxes(bboxes, img_width, img_height) for bboxes in train_annotations]
train_annotations = np.array(train_annotations, dtype=object)

# Check shapes of the training data
print(train_images.shape)  # (num_images, height, width, channels)
print(len(train_annotations))  # Should match the number of images

