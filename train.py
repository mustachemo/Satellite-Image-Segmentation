import logging
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import numpy as np
from pathlib import Path

from model.unet import build_unet_model
from utils.visualize import visualize_train_sample
from utils.logger_prep import get_logger
from utils.custom_funcs import dice_coefficient, combined_loss
from utils.directories_check import check_dirs, check_prepped_data
from configs import prepped_train_images, prepped_train_masks, prepped_test_images, prepped_test_masks





if __name__ == '__main__':

    logger = get_logger(__name__)
    
    gpu = tf.config.experimental.list_physical_devices('GPU')

    if not gpu:
        logger.warning('No GPU found, model may be slow or fail to train')
    else:
        logger.info(f'GPU found!')

    check_dirs()
    check_prepped_data()

    # Load the prepped data
    train_images = tf.convert_to_tensor(np.load(prepped_train_images))
    train_masks = tf.convert_to_tensor(np.load(prepped_train_masks))
    test_images = tf.convert_to_tensor(np.load(prepped_test_images))
    test_masks = tf.convert_to_tensor(np.load(prepped_test_masks))

    visualize_train_sample(train_images[4], train_masks[4])

    if Path('checkpoints/unet_model.h5').exists():
        logger.info('Model already exists, use predict.py to make predictions')
    else:
        logger.info('Model not found, creating and training...')
        model = build_unet_model()
        model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy', dice_coefficient])


        # Callbacks
        checkpoint = ModelCheckpoint('checkpoints/unet_model.h5', monitor='val_loss', save_best_only=True, mode='min')
        tensorboard = TensorBoard(log_dir='logs')
        csv_logger = CSVLogger('logs/training.log')

        # Train the model
        model.fit(train_images, train_masks, epochs=5, batch_size=1, validation_data=(test_images, test_masks), callbacks=[checkpoint, tensorboard, csv_logger])
        logger.info('Training complete')








    
