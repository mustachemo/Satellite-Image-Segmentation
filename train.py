from model.is_model import build_unet_model
from utils.data_loader_unet import load_and_process_files
from pathlib import Path
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import matplotlib.pyplot as plt
from utils.directories_check import check_dirs, check_prepped_data
import numpy as np
from configs import prepped_train_images, prepped_train_masks, prepped_test_images, prepped_test_masks
from utils.custom_funcs import dice_loss, dice_coefficient, combined_loss




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
    # fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # # Display the image and mask
    # for i in range(50):
    #     ax[0].imshow(train_images[i])
    #     ax[0].set_title('Image')

    #     ax[1].imshow(train_masks[i], cmap='gray')
    #     ax[1].set_title('Mask')


    #     plt.show()
    if Path('checkpoints/unet_model.h5').exists():
        logging.info('Model already exists, loading...')
        model = tf.keras.models.load_model('checkpoints/unet_model.h5', custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient})
    else:
        logging.info('Model not found, creating and training...')
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








    
