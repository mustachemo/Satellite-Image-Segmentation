from model.is_model import build_unet_model
from utils.data_loader_unet import load_and_process_files
from pathlib import Path
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import matplotlib.pyplot as plt
import os
import numpy as np

# Directory paths
train_images_dir = './data/images/train'
train_masks_dir = './data/mask/train'
test_images_dir = './data/images/val'
test_masks_dir = './data/mask/val'

# Prepped directories, images and masks resized to 256x256 and made into numpy arrays and grayscale
prepped_train_images = 'prepped_data/trainimages.npy'
prepped_train_masks = 'prepped_data/trainmasks.npy'
prepped_test_images = 'prepped_data/testimages.npy'
prepped_test_masks = 'prepped_data/testmasks.npy'

def check_dirs():
    '''Check if directories exist, if not create them'''
    if not os.path.exists(os.path.join('.', 'data')):
        os.mkdir(os.path.join('.', 'data'))
    if not os.path.exists(os.path.join('.', 'prepped_data')):
        os.mkdir(os.path.join('.', 'prepped_data'))
    if not os.path.exists(os.path.join('.', 'models')):
        os.mkdir(os.path.join('.', 'models'))
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
    

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.5 * dice + 0.5 * bce


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    gpu = tf.config.experimental.list_physical_devices('GPU')

    if not gpu:
        logging.warning('No GPU found, model may be slow or fail to train')
    else:
        logging.info(f'GPU found!')

    check_dirs()
    check_prepped_data()

    # Load the prepped data
    train_images = tf.convert_to_tensor(np.load(prepped_train_images))
    train_masks = tf.convert_to_tensor(np.load(prepped_train_masks))
    test_images = tf.convert_to_tensor(np.load(prepped_test_images))
    test_masks = tf.convert_to_tensor(np.load(prepped_test_masks))

    # Print an image and mask on the same plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Display the image and mask
    for i in range(50):
        ax[0].imshow(train_images[i])
        ax[0].set_title('Image')

        ax[1].imshow(train_masks[i], cmap='gray')
        ax[1].set_title('Mask')


        plt.show()
    
    model = build_unet_model()
    model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy', dice_coefficient])


    # Callbacks
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    checkpoint = ModelCheckpoint('checkpoints/unet_model.h5', monitor='val_loss', save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir='logs')
    csv_logger = CSVLogger('logs/training.log')

    # Train the model
    model.fit(train_images, train_masks, epochs=5, batch_size=1, validation_data=(test_images, test_masks), callbacks=[checkpoint, tensorboard, csv_logger])
    logging.info('Training complete')








    
