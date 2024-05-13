import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from pathlib import Path

print(Path.cwd())

# Load an example image and its mask
image_index = 'img_resize_404'
image_path = Path(r'data/images/val') / f'{image_index}.png'
mask_path = Path(r'data/mask/val') / f'{image_index}.png'
# Load the image and mask
try:
    image = Image.open(image_path)
    mask = Image.open(mask_path)
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit()

# Load bounding boxes from all_bbox.txt
try:
    with open(r'data\all_bbox.txt') as f:
        bboxes = json.load(f)
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit()

# Get bounding boxes for the current image
image_bboxes = bboxes.get(image_index, [])

# Create a figure and axis
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

# Display the image and mask
ax[0].imshow(image)
ax[0].set_title('Image')

ax[1].imshow(mask)
ax[1].set_title('Mask')

# Draw bounding boxes on the image
draw = ImageDraw.Draw(image)
for bbox in image_bboxes:
    draw.rectangle([bbox[2], bbox[3], bbox[0], bbox[1]], outline='yellow', width=3)

ax[0].imshow(image)

plt.show()
