import tensorflow as tf
import numpy as np

from utils.checker import check_dirs, check_prepped_data
from utils.custom_funcs import dice_loss, dice_coefficient, combined_loss
from utils.visualize import visaulize_prediction
from utils.logger_prep import get_logger
from configs import DROPOUT_RATE, ACTIVATION_FUNC

def prediction_for_single_model(test_dataset, activation_fun=ACTIVATION_FUNC):
    # Load the model
    try: 
        model = tf.keras.models.load_model(f'checkpoints/unet_model_{DROPOUT_RATE}_{activation_fun}.h5', custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient, 'combined_loss': combined_loss})
        logger.info('Model loaded successfully')
    except Exception as e:
        logger.error(f'Model not found, please train the model first: {e}')
        exit()

    # Predict and show results
    logger.info(f'Getting predictions for {len(test_dataset)} test samples')
    predictions = model.predict(test_dataset)
    
    # apply sigmoid to the predictions
    # predictions = tf.nn.sigmoid(predictions)
    
    for i in range(50, 60):
        visaulize_prediction(test_dataset[i][0], test_dataset[i][1], predictions[i])

    # Evaluate the model
    logger.info('Evaluating model')
    loss, accuracy, dice_coefficient_metric = model.evaluate(test_dataset)
    logger.info(f'Loss: {round(loss, 3)}, Accuracy: {round(accuracy, 3)}, Dice Coefficient: {round(dice_coefficient_metric, 3)}')
    
    logger.info('Predictions complete')

if __name__ == '__main__':

    check_dirs()
    dataset = check_prepped_data(get_train=False, get_test=True)
    logger = get_logger(__name__)


    # Predict for a single model
    prediction_for_single_model(dataset['test'])

    # Predict for multiple models
    # activation_funcs = ['relu', 'elu', 'swish', 'gelu', 'leaky_relu']
    # for activation_func in activation_funcs:
    #     prediction_for_single_model(test_images, test_masks, activation_func)
    #     logger.info(f'Prediction complete for model with activation function: {activation_func}')