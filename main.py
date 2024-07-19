import numpy as np
import tensorflow as tf
from configs import *
from utils.checker import check_dirs, check_prepped_data
from utils.custom_funcs import dice_loss, dice_coefficient, combined_loss
from utils.logger_prep import get_logger
from utils.visualize import visualize

from utils.MC_dropout import (
    mc_dropout_predictions,
    visualize_mean_std,
    visualize_confidence_intervals,
    plot_correlation_analysis,
    get_uncertainty_avgs,
    run_mc_dropout_on_all_images,
    visualize_mean_std_grid,
    visualize_mean_std_grid_multi_models,
)


if __name__ == "__main__":

    # visualize('421')

    logger = get_logger(__name__)

    check_dirs()
    dataset = check_prepped_data(get_train=False, get_test=True)

    #######################################################################
    # Uncertainty quantification using MC dropout #
    #######################################################################
    # Load the model and test data
    try:
        model = tf.keras.models.load_model(
            f"checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}.h5",
            custom_objects={
                "dice_loss": dice_loss,
                "dice_coefficient": dice_coefficient,
                "combined_loss": combined_loss,
            },
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model not found, please train the model first: {e}")
        exit()

    # Perform MC dropout inference on a single test image
    for i in range(50, 60):
        mc_predictions = mc_dropout_predictions(
            model, dataset["test"][i], num_samples=NUM_SAMPLES_MC_DROPOUT_PREDICTION
        )
        mean_prediction = np.mean(mc_predictions, axis=0)
        std_deviation = np.std(mc_predictions, axis=0)
        visualize_mean_std(
            dataset["test"][i], dataset["test"][1], mean_prediction, std_deviation
        )
        visualize_confidence_intervals(
            dataset["test"][i], mean_prediction, std_deviation, confidence_level=0.95
        )
        plot_correlation_analysis(mean_prediction, std_deviation)
        get_uncertainty_avgs(mean_prediction, std_deviation)

    # Perform MC dropout inference on all test images
    # mean_of_means, mean_of_stds = run_mc_dropout_on_all_images(model, test_images, num_samples=NUM_SAMPLES_MC_DROPOUT_PREDICTION)
    # plot_correlation_analysis(mean_of_means, mean_of_stds)
    # get_uncertainty_avgs(mean_of_means, mean_of_stds)

    # Visualize the mean and standard deviation for GRID_ITERATIONS number of test images
    predictions = np.array(
        [
            mc_dropout_predictions(
                model, dataset["test"][i], num_samples=NUM_SAMPLES_MC_DROPOUT_PREDICTION
            )
            for i in range(GRID_ITERATIONS)
        ]
    )
    visualize_mean_std_grid(
        dataset["test"][:GRID_ITERATIONS], predictions, rows=4, cols=GRID_ITERATIONS
    )
    logger.info("Uncertainty quantification for MC dropout complete")

    #######################################################################
    # Uncertainty quantification using MC dropout for multiple models (different activation functions) #
    #######################################################################
    # activation_funcs = ['relu', 'elu', 'swish', 'gelu', 'leaky_relu']
    # predictions_for_all_models = []
    # for activation_func in activation_funcs:
    #     try:
    #         model = tf.keras.models.load_model(f'checkpoints/unet_model_{DROPOUT_RATE}_{activation_func}.h5', custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient, 'combined_loss': combined_loss})
    #         logger.info('Model loaded successfully')
    #     except Exception as e:
    #         logger.error(f'Model not found, please train the model first: {e}')
    #         exit()

    #     # Perform MC dropout inference on a single image for each model
    #     mc_predictions = mc_dropout_predictions(model, test_images[0], num_samples=NUM_SAMPLES_MC_DROPOUT_PREDICTION)
    #     predictions_for_all_models.append(mc_predictions)

    # visualize_mean_std_grid_multi_models(test_images[0], test_masks[0], np.array(predictions_for_all_models), rows=4, cols=len(activation_funcs), activation_funcs=activation_funcs)
    # logger.info('Uncertainty quantification for multiple models complete')

    #######################################################################
    # Uncertainty quantification using MC dropout with ensemble of models #
    #######################################################################
    # combined_predictions = []
    # for model_num in range(5):
    #     try:
    #         model = tf.keras.models.load_model(f'checkpoints/unet_model_{DROPOUT_RATE}_{ACTIVATION_FUNC}_{model_num+1}.h5', custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient, 'combined_loss': combined_loss})
    #         logger.info('Model loaded successfully')
    #     except Exception as e:
    #         logger.error(f'Model not found, please train the model first: {e}')
    #         exit()

    #     # Perform MC dropout inference on a single image for each model
    #     mc_predictions = mc_dropout_predictions(model, test_images[0], num_samples=NUM_SAMPLES_MC_DROPOUT_PREDICTION)
    #     mc_predictions = np.mean(mc_predictions, axis=0)
    #     combined_predictions.append(mc_predictions)

    # mean_prediction = np.mean(combined_predictions, axis=0)
    # std_deviation = np.std(combined_predictions, axis=0)
    # visualize_mean_std(test_images[0], test_masks[0], mean_prediction, std_deviation)

    logger.info("Uncertainty quantification for ensemble of models complete")
