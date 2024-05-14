from tensorflow.keras.models import load_model
from utils.data_loader_unet import check_dirs, check_prepped_data
import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    check_dirs()
    check_prepped_data()

    # Load th
    test_images = tf.convert_to_tensor(np.load(prepped_test_images))
    test_masks = tf.convert_to_tensor(np.load(prepped_test_masks))

    # Load the model
    try: 
        model = load_model('checkpoints/unet_model.h5')
        logging.info('Model loaded successfully')
    except Exception as e:
        logging.error(f'Error: {e}')
        exit()

    # Predict and show results
    logging.info('Predicting test images')
    predictions = model.predict(test_images)
    for i in range(5):
        # Display the image and mask for prediction
        fig, ax = plt.subplots(1, 3, figsize=(15, 7))

        ax[0].imshow(test_images[i])
        ax[0].set_title('Image')

        ax[1].imshow(test_masks[i], cmap='gray')
        ax[1].set_title('Mask')

        ax[2].imshow(predictions[i], cmap='gray')
        ax[2].set_title('Predicted Mask')

        plt.show()