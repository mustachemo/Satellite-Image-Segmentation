from model.is_model import build_unet_model
from utils.data_loader_unet import train_images, train_masks, test_images, test_masks
from pathlib import Path
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras.metrics import MeanIoU
import matplotlib.pyplot as plt


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if not gpu:
        logging.warning('No GPU found, model may be slow or fail to train')
    else:
        logging.info(f'GPU found!')

    # Print an image and mask on the same plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Display the image and mask
    ax[0].imshow(train_images[4])
    ax[0].set_title('Image')

    ax[1].imshow(train_masks[4], cmap='gray')
    ax[1].set_title('Mask')

    plt.show()
    
    model = build_unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', MeanIoU(num_classes=2)])


    # Callbacks
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    checkpoint = ModelCheckpoint('checkpoints/unet_model.h5', monitor='val_loss', save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir='logs')
    csv_logger = CSVLogger('logs/training.log')

    # Train the model
    model.fit(train_images, train_masks, epochs=5, batch_size=1, validation_data=(test_images, test_masks), callbacks=[checkpoint, tensorboard, csv_logger])
    logging.info('Training complete')

    # Predict and show results
    logging.info('Predicting test images')
    predictions = model.predict(test_images)
    for i in range(5):
        plt.imshow(test_images[i])
        plt.imshow(predictions[i])
        plt.show()






    
