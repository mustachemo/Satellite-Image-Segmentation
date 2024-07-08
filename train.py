import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, LearningRateScheduler # type: ignore
import numpy as np
from pathlib import Path

from model.unet import build_unet_model
from model.bayesian_unet import build_bayesian_unet_model
from utils.visualize import visualize_sample_with_mask
from utils.logger_prep import get_logger
from utils.custom_funcs import dice_coefficient, combined_loss, combined_loss_bayesian_unet, uncertainty_aware_loss
from utils.checker import check_dirs, check_prepped_data
from utils.utils import calculate_memory_usage
from configs import *

logger = get_logger(__name__)

def train_unet(train_dataset, test_dataset):
    if Path(f'checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.keras').exists():
        logger.info('Model already exists, exiting...')
        return

    tf.profiler.experimental.start(logdir='/logs/profiler/')
    if MIXED_PRECISION:
        policy = tf.keras.mixed_precision.Policy('mixed_float16') # mixed precisions means using 16-bit floating point numbers where possible
        tf.keras.mixed_precision.set_global_policy(policy)
    
    logger.info('Model not found, creating and training...')
    model = build_unet_model(dropout_rate=DROPOUT_RATE)
    model.compile(optimizer='adam', loss=combined_loss, metrics=[dice_coefficient])

    # Callbacks
    # montior dice coefficient
    checkpoint = ModelCheckpoint(f'checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.keras', monitor='val_dice_coefficient', save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir='logs')
    csv_logger = CSVLogger(f'logs/model_{DROPOUT_RATE}_{ACTIVATION_FUNC}_training.log')
    learning_rate_scheduler = LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/10))

    # Train the model
    model.fit(train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_dataset), callbacks=[checkpoint, tensorboard, csv_logger, learning_rate_scheduler], verbose=2)
    logger.info('Training complete for UNet model')
    tf.profiler.experimental.stop()

def train_bayesian_unet(train_dataset, test_dataset):
    if Path(f'checkpoints/bayesian_unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.h5').exists():
        logger.info('Model already exists, exiting...')
        pass

    logger.info('Model not found, creating and training...')
    model = build_bayesian_unet_model(dropout_rate=DROPOUT_RATE)
    model.compile(optimizer='adam', loss=combined_loss_bayesian_unet, metrics=['accuracy', dice_coefficient])

    # Callbacks
    checkpoint = ModelCheckpoint(f'checkpoints/bayesian_unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.h5', monitor='val_dice_coefficient', save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir='logs')
    csv_logger = CSVLogger('logs/training.log')

    # Train the model
    model.fit(train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_dataset), callbacks=[checkpoint, tensorboard, csv_logger])
    logger.info('Training complete for Bayesian UNet model')


if __name__ == '__main__':
    # make sure the tensorflow version is 2.16.1
    if tf.__version__ != '2.16.2':
        logger.warning(f'You are using TensorFlow version {tf.__version__}, this code is tested with TensorFlow version 2.16.2')
        logger.warning('The code may not work with other versions of TensorFlow')
    
    
    check_dirs()
    dataset = dict()
    dataset = check_prepped_data(get_train=True, get_test=True)

    gpu = tf.config.experimental.list_physical_devices('GPU')

    if not gpu:
        logger.warning('No GPU found, model may be slow or fail to train')
    else:
        logger.info(f'GPU found!')
        # Allow memory growth to avoid OOM errors
        # This is not the best way to handle memory issues, but it works for now
        # For better memory management, use tf.data API and tf.data.Dataset.cache() method
        # https://www.tensorflow.org/guide/data_performance#prefetching
        # ! This will not work with the current implementation of the data loader
        # ! The data loader will need to be re-implemented using tf.data API
        # ! This is a TODO
        # This tells TensorFlow to allocate only as much GPU memory as needed
        for device in gpu:
            tf.config.experimental.set_memory_growth(device, True)

    
    # Check if the data is loaded correctly and normalized
    for sample_image, sample_mask in dataset['train'].take(1):
        # Select the first image from the batch
        sample_image = sample_image.numpy()[0, :, :, :]
        sample_mask = sample_mask.numpy()[0, :, :, :]

        visualize_sample_with_mask(sample_image, sample_mask)
        logger.info(f'Min and max values of image: {np.min(sample_image), np.max(sample_mask)}')
        logger.info(f'Min and max values of mask: {np.min(sample_mask), np.max(sample_mask)}')
        print(f'Memory usage per batch: {calculate_memory_usage(sample_image, sample_mask)} MB')
        print('--'*20)


    # Train the UNet model
    train_unet(dataset['train'], dataset['test'])

    # Train the Bayesian UNet model
    #! Does not work currently due to memory issues
    # train_bayesian_unet(dataset['train'], dataset['test'])









    
