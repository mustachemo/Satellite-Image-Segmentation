import numpy as np
import tensorflow as tf
from configs import *
from utils.directories_check import check_dirs, check_prepped_data
from utils.custom_funcs import dice_loss, dice_coefficient, combined_loss
from utils.logger_prep import get_logger
from utils.visualize import visualize

from uncertainty_quantification.MC_dropout import (
    mc_dropout_predictions, 
    visualize_mean_std, 
    visualize_confidence_intervals, 
    plot_correlation_analysis, 
    get_uncertainty_avgs, 
    run_mc_dropout_on_all_images, 
    visualize_mean_std_grid
)

if __name__ == '__main__':

    # visualize('421')

    logger = get_logger(__name__)

    # Load the model and test data
    try: 
        model = tf.keras.models.load_model(f'checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.h5', custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient, 'combined_loss': combined_loss})
        logger.info('Model loaded successfully')
    except Exception as e:
        logger.error(f'Model not found, please train the model first: {e}')
        exit()

    check_dirs()
    check_prepped_data()

    test_images = tf.convert_to_tensor(np.load(PREPPED_TEST_IMAGES))
    test_masks = tf.convert_to_tensor(np.load(PREPPED_TEST_MASKS))


    #######################################################################
    # Uncertainty quantification using MC dropout #
    #######################################################################
    # Perform MC dropout inference on a single test image
    mc_predictions = mc_dropout_predictions(model, test_images[0], num_samples=NUM_SAMPLES_MC_DROPOUT_PREDICTION)
    mean_prediction = np.mean(mc_predictions, axis=0)
    std_deviation = np.std(mc_predictions, axis=0)
    visualize_mean_std(test_images[0], test_masks[0], mean_prediction, std_deviation)
    visualize_confidence_intervals(test_images[0], mean_prediction, std_deviation, confidence_level=0.95)
    plot_correlation_analysis(mean_prediction, std_deviation)
    get_uncertainty_avgs(mean_prediction, std_deviation)

    # Perform MC dropout inference on all test images
    mean_of_means, mean_of_stds = run_mc_dropout_on_all_images(model, test_images, num_samples=NUM_SAMPLES_MC_DROPOUT_PREDICTION)
    plot_correlation_analysis(mean_of_means, mean_of_stds)
    get_uncertainty_avgs(mean_of_means, mean_of_stds)

    # Visualize the mean and standard deviation for GRID_ITERATIONS number of test images
    predictions = np.array([mc_dropout_predictions(model, test_image, num_samples=NUM_SAMPLES_MC_DROPOUT_PREDICTION) for test_image in test_images[:GRID_ITERATIONS]])
    visualize_mean_std_grid(test_images[:GRID_ITERATIONS], test_masks[:GRID_ITERATIONS], predictions, rows=4, cols=GRID_ITERATIONS)

    #######################################################################

    logger.info('Uncertainty quantification experiment complete')