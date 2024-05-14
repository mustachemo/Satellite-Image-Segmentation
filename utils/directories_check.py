import os
from pathlib import Path
import logging
from utils.data_loader_unet import load_and_process_files
from configs import train_images_dir, train_masks_dir, test_images_dir, test_masks_dir, prepped_train_images, prepped_train_masks, prepped_test_images, prepped_test_masks

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
    if not Path(prepped_train_images).exists():
        logging.info('Prepped training data not found, creating...')
        load_and_process_files(train_images_dir, train_masks_dir, prefix='train')
    if not Path(prepped_test_images).exists():
        logging.info('Prepped test data not found, creating...')
        load_and_process_files(test_images_dir, test_masks_dir, prefix='test')