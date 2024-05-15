import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from configs import PREPPED_TEST_IMAGES, PREPPED_TEST_MASKS, UQ_TEST_EXAMPLE_INDEX, DROPOUT_RATE
from utils.directories_check import check_dirs, check_prepped_data
from utils.custom_funcs import dice_loss, dice_coefficient, combined_loss
from utils.logger_prep import get_logger

from uncertainity_quantification.MC_dropout import mc_dropout_predictions, visualize_mean_std


if __name__ == '__main__':

    logger = get_logger(__name__)

    # Load the model and test data
    try: 
        model = tf.keras.models.load_model(f'checkpoints/unet_model_{DROPOUT_RATE}.h5', custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient, 'combined_loss': combined_loss})
        logger.info('Model loaded successfully')
    except Exception as e:
        logger.error(f'Model not found, please train the model first: {e}')
        exit()

    check_dirs()
    check_prepped_data()

    test_images = tf.convert_to_tensor(np.load(PREPPED_TEST_IMAGES))
    test_masks = tf.convert_to_tensor(np.load(PREPPED_TEST_MASKS))

    # Uncertainty quantification using MC dropout
    test_image = np.expand_dims(test_images[UQ_TEST_EXAMPLE_INDEX], axis=0)
    test_mask = np.expand_dims(test_masks[UQ_TEST_EXAMPLE_INDEX], axis=0)
    mc_predictions = mc_dropout_predictions(model, test_image, num_samples=100)
    # Calculate mean and standard deviation
    mean_prediction = np.mean(mc_predictions, axis=0)
    std_deviation = np.std(mc_predictions, axis=0)

    visualize_mean_std(test_image, test_mask, mean_prediction, std_deviation)
   

    logger.info('Uncertainty quantification experiment complete')