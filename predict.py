from utils.directories_check import check_dirs, check_prepped_data
from utils.custom_funcs import dice_loss, dice_coefficient, combined_loss
import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt
from configs import prepped_test_images, prepped_test_masks

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    check_dirs()
    check_prepped_data()

    # Load the model
    try: 
        model = tf.keras.models.load_model('checkpoints/unet_model.h5', custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient, 'combined_loss': combined_loss})
        logging.info('Model loaded successfully')
    except Exception as e:
        logging.error(f'Model not found, please train the model first: {e}')
        exit()


    test_images = tf.convert_to_tensor(np.load(prepped_test_images))
    test_masks = tf.convert_to_tensor(np.load(prepped_test_masks))

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