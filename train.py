from model.od_model import model
from pathlib import Path
from PIL import Image
import numpy as np
import json

# Define the directories where your images and annotations are stored
image_dir = Path('data/images/train')
annotation_dir = Path('data/mask/train')

# Load your training images
train_images = []
for image_path in image_dir.glob('*.png'):  # Adjust the file extension as needed
    image = Image.open(image_path)
    image = np.array(image)
    train_images.append(image)
train_images = np.array(train_images)

# Load your training annotations
train_annotations = []
for annotation_path in annotation_dir.glob('*.json'):  # Adjust the file extension as needed
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    train_annotations.append(annotation)
train_annotations = np.array(train_annotations)

# Check shapes of the training data
print(train_images.shape)  # (num_images, height, width, channels)
print(train_annotations.shape)  # (num_images, num_annotations, 5)

# Train the model
num_epochs = 10  # Adjust the number of epochs as needed
batch_size = 32  # Adjust the batch size as needed

model.fit(train_images, train_annotations, epochs=num_epochs, batch_size=batch_size)

# Save the trained model
model.save('object_detection_model.h5')