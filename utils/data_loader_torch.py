import torch
import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import numpy as np

# Configuration values
X_DIMENSION = 256
Y_DIMENSION = 256
BATCH_SIZE = 32
BUFFER_SIZE = 1000
MIXED_PRECISION = False

def process_image(image_path, mask_path, make_mask_binary=True):
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Grayscale

        image = cv2.resize(image, (X_DIMENSION, Y_DIMENSION), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (X_DIMENSION, Y_DIMENSION), interpolation=cv2.INTER_LINEAR)

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # Convert to PyTorch tensor and normalize to [0, 1]
        mask = torch.tensor(mask).unsqueeze(0).float() / 255.0  # Convert to PyTorch tensor and normalize

        if make_mask_binary:
            mask = torch.where(mask > 0.5, torch.tensor(1.0), torch.tensor(0.0)).type(torch.uint8)
        
        return image, mask
    except:
        return None, None

def load_and_process_files(image_dir, mask_dir, prefix="train"):
    images = []
    masks = []
    image_paths = [f for f in image_dir.glob("*.png")]

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image_path, mask_dir / f"{image_path.stem}_mask.png"): image_path for image_path in image_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading and processing {prefix} data"):
            image, mask = future.result()
            if image is not None and mask is not None:
                images.append(image)
                masks.append(mask)

    return images, masks

def save_tensor(filename, images, masks):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(torch.save, (image, mask), os.path.join(filename, f"./prepped_data/images/train/datapoint_{i}.pt"))
                   for i, (image, mask) in enumerate(zip(images, masks))]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Saving Tensors"):
            future.result()

def load_tensor(filename):
    tensors = []
    for tensor_file in os.listdir(filename):
        if tensor_file.endswith(".pt"):
            tensors.append(torch.load(os.path.join(filename, tensor_file)))
    return tensors

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list
    
    def __len__(self):
        return len(self.tensor_list)
    
    def __getitem__(self, idx):
        image, mask = self.tensor_list[idx]
        if MIXED_PRECISION:
            image = image.half()
            mask = mask.half()
        return image, mask

def create_pytorch_dataloader(tensor_list, batch_size=BATCH_SIZE):
    dataset = CustomDataset(tensor_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
