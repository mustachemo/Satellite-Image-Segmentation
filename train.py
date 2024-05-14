from model.is_model import build_unet_model
from utils.data_loader_unet import train_images, train_masks, test_images, test_masks
from pathlib import Path
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
import matplotlib.pyplot as plt


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if not gpu:
        logging.warning('No GPU found, model may be slow or fail to train')
    else:
        logging.info(f'GPU found!')

    
    model = build_unet_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    checkpoint = ModelCheckpoint('checkpoints/unet_model.h5', save_best_only=True)
    tensorboard = TensorBoard(log_dir='logs')
    csv_logger = CSVLogger('logs/training.log')

    # Train the model
    model.fit(train_images, train_masks, epochs=5, batch_size=1, validation_data=(test_images, test_masks), callbacks=[reduce_lr, checkpoint, tensorboard, csv_logger])
    logging.info('Training complete')

    # Predict and show results
    logging.info('Predicting test images')
    predictions = model.predict(test_images)
    for i in range(5):
        plt.imshow(test_images[i])
        plt.imshow(predictions[i])
        plt.show()






    
