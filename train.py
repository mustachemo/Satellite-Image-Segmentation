import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import numpy as np
from pathlib import Path

from model.unet import build_unet_model
from model.bayesian_unet import build_bayesian_unet_model
from utils.visualize import visualize_train_sample
from utils.logger_prep import get_logger
from utils.custom_funcs import dice_coefficient, combined_loss, combined_loss_bayesian_unet, uncertainty_aware_loss
from utils.directories_check import check_dirs, check_prepped_data
from utils.data_loader_unet_tf import load_and_process_files
from configs import *

logger = get_logger(__name__)

def train_unet(train_images, train_masks, test_images, test_masks):
    if Path(f'checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.h5').exists():
        logger.info('Model already exists, exiting...')
        pass
    else:
        logger.info('Model not found, creating and training...')
        model = build_unet_model(dropout_rate=DROPOUT_RATE)

        optimizer = tf.keras.optimizers.Adam()
        # optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, name='SGD')
        model.compile(optimizer=optimizer, loss=combined_loss, metrics=['accuracy', dice_coefficient])

        # Callbacks
        # montior dice coefficient
        checkpoint = ModelCheckpoint(f'checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.h5', monitor='val_dice_coefficient', save_best_only=True, mode='max')
        tensorboard = TensorBoard(log_dir='logs')
        csv_logger = CSVLogger(f'logs/model_{DROPOUT_RATE}_{ACTIVATION_FUNC}_training.log')

    
        train_dataset = load_and_process_files(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, prefix='train')
        for item in train_dataset.take(5):
            # show the image and mask
            visualize_train_sample(item[0][0], item[1][0])
        test_dataset = load_and_process_files(TEST_IMAGES_DIR, TEST_MASKS_DIR,  prefix='test')
        # Train the model
        # model.fit(train_images, train_masks, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_images, test_masks), callbacks=[checkpoint, tensorboard, csv_logger])
        model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, callbacks=[checkpoint, tensorboard, csv_logger])
        logger.info('Training complete for UNet model')

def train_bayesian_unet(train_images, train_masks, test_images, test_masks):
    if Path(f'checkpoints/bayesian_unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.h5').exists():
        logger.info('Model already exists, exiting...')
        pass
    else:
        logger.info('Model not found, creating and training...')
        model = build_bayesian_unet_model(dropout_rate=DROPOUT_RATE)
        model.compile(optimizer='adam', loss=combined_loss_bayesian_unet, metrics=['accuracy', dice_coefficient])

        # Callbacks
        checkpoint = ModelCheckpoint(f'checkpoints/bayesian_unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.h5', monitor='val_dice_coefficient', save_best_only=True, mode='max')
        tensorboard = TensorBoard(log_dir='logs')
        csv_logger = CSVLogger('logs/training.log')

        # Train the model
        model.fit(train_images, train_masks, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_images, test_masks), callbacks=[checkpoint, tensorboard, csv_logger])
        logger.info('Training complete for Bayesian UNet model')




if __name__ == '__main__':

    check_dirs()
    check_prepped_data()

    gpu = tf.config.experimental.list_physical_devices('GPU')

    if not gpu:
        logger.warning('No GPU found, model may be slow or fail to train')
    else:
        logger.info(f'GPU found!')

    # Load the prepped data
    train_images = tf.convert_to_tensor(np.load(PREPPED_TRAIN_IMAGES))
    train_masks = tf.convert_to_tensor(np.load(PREPPED_TRAIN_MASKS))
    test_images = tf.convert_to_tensor(np.load(PREPPED_TEST_IMAGES))
    test_masks = tf.convert_to_tensor(np.load(PREPPED_TEST_MASKS))

    # visualize_train_sample(train_images[4], train_masks[4])

    # Train the UNet model
    train_unet(train_images, train_masks, test_images, test_masks)

    # Train the Bayesian UNet model
    #! Does not work currently due to memory issues
    # train_bayesian_unet(train_images, train_masks, test_images, test_masks)









    
