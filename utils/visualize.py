import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from pathlib import Path

def load_image_and_mask(image_index):
    """
    Load image and mask based on the image index.
    """
    image_path = Path(r'data/images/val') / f'img_resize_{image_index}.png'
    mask_path = Path(r'data/mask/val') / f'img_resize_{image_index}_mask.png'

    try:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        exit()

    return image, mask

def load_bboxes():
    """
    Load bounding boxes from all_bbox.txt.
    """
    try:
        with open(r'data\all_bbox.txt') as f:
            bboxes = json.load(f)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        exit()

    return bboxes

def draw_bboxes_on_image(image, bboxes):
    """
    Draw bounding boxes on the image.
    """
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle([bbox[2], bbox[3], bbox[0], bbox[1]], outline='yellow', width=3)

    return image

def visualize(image_index):
    """
    Visualize image, mask and bounding boxes.
    """
    image, mask = load_image_and_mask(image_index)
    bboxes = load_bboxes()
    image_bboxes = bboxes.get(image_index, [])

    # Create a figure and axis
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Display the image and mask
    ax[0].imshow(image)
    ax[0].set_title('Image + Bounding Box')

    ax[1].imshow(mask)
    ax[1].set_title('Mask')

    # Draw bounding boxes on the image
    image = draw_bboxes_on_image(image, image_bboxes)

    ax[0].imshow(image)

    plt.show()

def visualize_train_sample(train_images, train_masks):
    """
    Visualize a sample image, mask and bounding boxes.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    ax[0].imshow(train_images)
    ax[0].set_title('Image')
    ax[1].imshow(train_masks, cmap='gray')
    ax[1].set_title('Mask')

    plt.show()

def visualize_test_sample(test_images, test_masks, predictions):
    """
    Visualize a sample image, mask and prediction.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 7))

    ax[0].imshow(test_images)
    ax[0].set_title('Image')
    ax[1].imshow(test_masks, cmap='gray')
    ax[1].set_title('Mask')
    ax[2].imshow(predictions, cmap='gray')
    ax[2].set_title('Predicted Mask')

    plt.show()