from model.is_model import build_unet_model
# from utils.data_loader_unet import DatasetLoader
from utils.attempt1 import create_dataset
from utils.attempt2 import images_train, images_val, masks_train, masks_val
from pathlib import Path
import logging
import tensorflow as tf

if __name__ == '__main__':
    image_dir = Path('./temp_data/images/train')
    mask_dir = Path('./temp_data/mask/train')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if not gpu:
        logging.warning('No GPU found, model may be slow or fail to train')
    else:
        logging.info(f'GPU found!')

    # Load the dataset
    # dataset = DatasetLoader(images_dir=image_dir, mask_dir=mask_dir, scale=1.0)

    image_dir = './data/images/train'
    mask_dir = './data/mask/train'
    train_dataset = create_dataset(image_dir, mask_dir)
    for images, masks in train_dataset.take(2):
        print("Image batch shape: ", images.shape)
        print("Mask batch shape: ", masks.shape)
    
    model = build_unet_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(images_train, masks_train, epochs=5, batch_size=1, validation_data=(images_val, masks_val))

    # Save the model
    model.save('unet_model.h5')
