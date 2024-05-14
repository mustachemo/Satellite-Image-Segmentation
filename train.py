from model.is_model import build_unet_model
from utils.data_loader_unet import DatasetLoader
from pathlib import Path
import logging
import tensorflow as tf

if __name__ == '__main__':
    image_dir = Path(r'./temp_data/images/train')
    mask_dir = Path(r'./temp_data/mask/train')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if not gpu:
        logging.warning('No GPU found, model may be slow or fail to train')
    else:
        logging.info(f'GPU found!')

    # Load the dataset
    dataset = DatasetLoader(images_dir=image_dir, mask_dir=mask_dir, scale=1.0)
    model = build_unet_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(dataset, epochs=5)

    # Save the model
    model.save('unet_model.h5')
