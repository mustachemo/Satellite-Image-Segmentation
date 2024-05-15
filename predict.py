import tensorflow as tf
import numpy as np
import logging

from utils.directories_check import check_dirs, check_prepped_data
from utils.custom_funcs import dice_loss, dice_coefficient, combined_loss
from utils.visualize import visualize_test_sample
from utils.logger_prep import get_logger
from configs import PREPPED_TEST_IMAGES, PREPPED_TEST_MASKS, DROPOUT_RATE

if __name__ == '__main__':

    check_dirs()
    check_prepped_data()
    logger = get_logger(__name__)

    # Load the model
    try: 
        model = tf.keras.models.load_model(f'checkpoints/unet_model_{DROPOUT_RATE}.h5', custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient, 'combined_loss': combined_loss})
        logger.info('Model loaded successfully')
    except Exception as e:
        logger.error(f'Model not found, please train the model first: {e}')
        exit()


    test_images = tf.convert_to_tensor(np.load(PREPPED_TEST_IMAGES))
    test_masks = tf.convert_to_tensor(np.load(PREPPED_TEST_MASKS))

    # Predict and show results
    logger.info('Predicting test images')
    predictions = model.predict(test_images)

    for i in range(1):
        visualize_test_sample(test_images[i], test_masks[i], predictions[i])

    # Evaluate the model
    logger.info('Evaluating model')
    loss, accuracy, dice_coefficient_metric = model.evaluate(test_images, test_masks)
    logger.info(f'Loss: {round(loss, 3)}, Accuracy: {round(accuracy, 3)}, Dice Coefficient: {round(dice_coefficient_metric, 3)}')
    
    logger.info('Predictions complete')