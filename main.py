import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from configs import prepped_test_images, prepped_test_masks
from utils.directories_check import check_dirs, check_prepped_data
from utils.custom_funcs import dice_loss, dice_coefficient, combined_loss
from utils.logger_prep import get_logger

from uncertainity_quantification.MC_dropout import mc_dropout_predictions

if __name__ == '__main__':
    check_dirs()
    check_prepped_data()
    logger = get_logger(__name__)

    # Load the model and test data
    try: 
        model = tf.keras.models.load_model('checkpoints/unet_model.h5', custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient, 'combined_loss': combined_loss})
        logger.info('Model loaded successfully')
    except Exception as e:
        logger.error(f'Model not found, please train the model first: {e}')
        exit()

    test_images = tf.convert_to_tensor(np.load(prepped_test_images))
    test_masks = tf.convert_to_tensor(np.load(prepped_test_masks))

    # Uncertainty quantification using MC dropout
    test_image = np.expand_dims(test_images[0], axis=0)
    mc_predictions = mc_dropout_predictions(model, test_image, num_samples=100)
    # Calculate mean and standard deviation
    mean_prediction = np.mean(mc_predictions, axis=0)
    std_deviation = np.std(mc_predictions, axis=0)

    # Visualize the uncertainty as a normal distribution
    plt.figure(figsize=(10, 10))
    
    # Display the actual test image
    plt.subplot(2, 2, 1)
    plt.imshow(test_image[0, :, :, 0])  # Adjust indices according to your data shape
    plt.colorbar()
    plt.title('Test Image')
    
    # Display the actual mask
    plt.subplot(2, 2, 2)
    plt.imshow(test_masks[0, :, :, 0], cmap='gray')  # Adjust indices according to your data shape
    plt.colorbar()
    plt.title('Test Mask')
    
    # Display the mean prediction
    plt.subplot(2, 2, 3)
    plt.imshow(mean_prediction[0, :, :, 0], cmap='gray')  # Adjust indices according to your data shape
    plt.colorbar()
    plt.title('Mean Prediction')
    
    # Display the prediction uncertainty
    plt.subplot(2, 2, 4)
    plt.imshow(std_deviation[0, :, :, 0], cmap='gray')  # Adjust indices according to your data shape
    plt.colorbar()
    plt.title('Prediction Uncertainty (Standard Deviation)')
    
    plt.show()


    logger.info('Uncertainty quantification complete')