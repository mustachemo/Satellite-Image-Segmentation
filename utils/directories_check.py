import logging
from utils.data_loader_unet import load_and_process_files
from configs import TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, TEST_IMAGES_DIR, TEST_MASKS_DIR, PREPPED_TRAIN_IMAGES, PREPPED_TEST_IMAGES
from pathlib import Path

def check_dirs():
    '''Check if directories exist, if not create them'''
    directories = ['checkpoints', 'data', 'prepped_data', 'model', 'logs']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def check_prepped_data():
    '''Check if prepped data exists, if not create it'''
    paths = {
        'train': (PREPPED_TRAIN_IMAGES, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR),
        'test': (PREPPED_TEST_IMAGES, TEST_IMAGES_DIR, TEST_MASKS_DIR),
    }
    
    for prefix, (prepped_path, images_dir, masks_dir) in paths.items():
        if not Path(prepped_path).exists():
            logging.info(f'Prepped {prefix} data not found, creating...')
            load_and_process_files(Path(images_dir), Path(masks_dir), prefix=prefix)