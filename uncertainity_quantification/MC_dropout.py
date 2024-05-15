import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def mc_dropout_predictions(model, input_image, num_samples=50):
    """
    Perform MC dropout inference to get a distribution of predictions.
    
    Args:
    model (tf.keras.Model): Trained model with dropout layers.
    input_image (numpy.array): Input image for which predictions are needed.
    num_samples (int): Number of stochastic forward passes.
    
    Returns:
    numpy.array: Array of predictions from each forward pass.
    """
    predictions = [model(input_image, training=True) for _ in tqdm(range(num_samples), desc='MC Dropout Inference')]
    predictions = np.array(predictions)
    return predictions


def visualize_mean_std(test_image, test_mask, mean_prediction, std_deviation):
    """ Visualize the test image, mask and model predictions. """
    plt.figure(figsize=(10, 10))
    
    # Display the actual test image
    plt.subplot(2, 2, 1)
    plt.imshow(test_image[0, :, :, :])
    plt.colorbar()
    plt.title('Test Image')
    
    # Display the actual mask
    plt.subplot(2, 2, 2)
    plt.imshow(test_mask[0, :, :, 0], cmap='viridis')
    plt.colorbar()
    plt.title('Test Mask')
    
    # Display the mean prediction
    plt.subplot(2, 2, 3)
    plt.imshow(mean_prediction[0, :, :, 0], cmap='viridis')
    plt.colorbar()
    plt.title('Mean Prediction')
    
    # Display the prediction uncertainty
    plt.subplot(2, 2, 4)
    plt.imshow(std_deviation[0, :, :, 0], cmap='viridis')
    plt.colorbar()
    plt.title('Prediction Uncertainty (std)')
    
    plt.show()


# Confidence Interval Visualizations
def visualize_confidence_intervals(test_image, mean_prediction, std_deviation, confidence_level=0.95):
    """ Visualize the confidence intervals of the model predictions.
        95% lower-bound means that 95% of the predictions will be above this value.
        95% upper-bound means that 95% of the predictions will be below this value.
    """
    # Calculate the confidence intervals
    lower_bound = mean_prediction - std_deviation * 1.96
    upper_bound = mean_prediction + std_deviation * 1.96
    
    # Display the actual test image
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.imshow(test_image[0, :, :, :])
    plt.colorbar()
    plt.title('Test Image')
    
    # Display the mean prediction
    plt.subplot(2, 2, 2)
    plt.imshow(mean_prediction[0, :, :, 0], cmap='viridis')
    plt.colorbar()
    plt.title('Mean Prediction')
    
    # Display the lower bound of the confidence interval
    plt.subplot(2, 2, 3)
    plt.imshow(lower_bound[0, :, :, 0], cmap='viridis')
    plt.colorbar()
    plt.title(f'{confidence_level * 100}% Confidence Lower Bound')
    
    # Display the upper bound of the confidence interval
    plt.subplot(2, 2, 4)
    plt.imshow(upper_bound[0, :, :, 0], cmap='viridis')
    plt.colorbar()
    plt.title(f'{confidence_level * 100}% Confidence Upper Bound')
    
    plt.show()


