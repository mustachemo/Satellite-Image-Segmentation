import os
from pathlib import Path
import logging
from utils.data_loader_unet import load_and_process_files
from configs import TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, TEST_IMAGES_DIR, TEST_MASKS_DIR, PREPPED_TRAIN_IMAGES, PREPPED_TEST_IMAGES

def check_dirs():
    '''Check if directories exist, if not create them'''
    if not os.path.exists(os.path.join('.', 'data')):
        os.mkdir(os.path.join('.', 'data'))
    if not os.path.exists(os.path.join('.', 'prepped_data')):
        os.mkdir(os.path.join('.', 'prepped_data'))
    if not os.path.exists(os.path.join('.', 'model')):
        os.mkdir(os.path.join('.', 'model'))
    if not os.path.exists(os.path.join('.', 'logs')):
        os.mkdir(os.path.join('.', 'logs'))

def check_prepped_data():
    '''Check if prepped data exists, if not create it'''
    if not Path(PREPPED_TRAIN_IMAGES).exists():
        logging.info('Prepped training data not found, creating...')
        load_and_process_files(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, prefix='train')
    if not Path(PREPPED_TEST_IMAGES).exists():
        logging.info('Prepped test data not found, creating...')
        load_and_process_files(TEST_IMAGES_DIR, TEST_MASKS_DIR, prefix='test')