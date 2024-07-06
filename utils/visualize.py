import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
from .custom_funcs import dice_coefficient
import pandas as pd
import glob


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

def visualize_sample_with_mask(image, mask):
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
    plt.close()

def visaulize_prediction(test_image, test_mask, prediction):
    """
    Visualize a sample image, mask and prediction.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 7))

    ax[0].imshow(test_image)
    ax[0].set_title('Image')
    ax[1].imshow(test_mask, cmap='gray')
    ax[1].set_title('Mask')
    ax[2].imshow(prediction, cmap='gray')
    ax[2].set_title('Predicted Mask')

    fig.text(0.5, 0.1, f'Dice Coefficient: {dice_coefficient(test_mask, prediction):.4f}', ha='center', fontsize=12)

    plt.show()


def visualize_training_logs():
    log_files = glob.glob('logs/model_0.35_relu_training_lambda_*.log')
    data = {}

    for log_file in log_files:

        lambda_value = log_file.split('_')[-1].split('.log')[0]
        
        df = pd.read_csv(log_file)
        
        data[lambda_value] = df['val_dice_coefficient']

    plt.figure(figsize=(10, 6))

    for lambda_value, val_dice_coefs in data.items():
        plt.plot(val_dice_coefs, label=f'lambda={lambda_value}')

    plt.xlabel('Epoch')
    plt.ylabel('Validation Dice Coefficient')
    plt.title('Validation Dice Coefficient vs Epoch for Different Lambda Values')
    plt.legend(title='Lambda Values')
    plt.grid(True)
    plt.show()