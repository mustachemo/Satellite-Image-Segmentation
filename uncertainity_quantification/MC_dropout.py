import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.logger_prep import get_logger

logger = get_logger(__name__)

def run_mc_dropout_on_all_images(model, images, num_samples=10):
    """
    Run MC dropout prediction on all test images multiple times and average the results.

    Args:
    model: The loaded TensorFlow model with dropout.
    images: A batch of images (num_images, height, width, channels).
    num_samples: Number of Monte Carlo samples to generate per image.

    Returns:
    Tuple of mean predictions and standard deviations for all images.
    """
    all_mean_predictions = []
    all_std_deviations = []

    # Loop through each image
    for image in tqdm(images, desc=f'MC Dropout Inference for all test images ({num_samples} times each)'):
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        mc_samples = [model(image, training=True) for _ in range(num_samples)]
        mc_samples = np.array(mc_samples)

        
        # Calculate mean and standard deviation
        mean_prediction = np.mean(mc_samples, axis=0)
        std_deviation = np.std(mc_samples, axis=0)
        
        all_mean_predictions.append(mean_prediction)
        all_std_deviations.append(std_deviation)
    
    # Average over all images
    mean_of_means = np.mean(all_mean_predictions, axis=0)
    mean_of_stds = np.mean(all_std_deviations, axis=0)

    return mean_of_means, mean_of_stds

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
    input_image = np.expand_dims(input_image, axis=0)
    predictions = [model(input_image, training=True) for _ in tqdm(range(num_samples), desc='MC Dropout Inference for a single test image')]
    predictions = np.array(predictions)
    return predictions


def visualize_mean_std(test_image, test_mask, mean_prediction, std_deviation):
    """ Visualize the test i mage, mask and model predictions. """
    plt.figure(figsize=(10, 10))
    
    # Display the actual test image
    plt.subplot(2, 2, 1)
    plt.imshow(test_image[:, :, :])
    plt.colorbar()
    plt.title('Test Image')
    
    # Display the actual mask
    plt.subplot(2, 2, 2)
    plt.imshow(test_mask[:, :, 0], cmap='gray')
    plt.colorbar()
    plt.title('Test Mask')
    
    # Display the mean prediction
    plt.subplot(2, 2, 3)
    plt.imshow(mean_prediction[0, :, :, 0], cmap='gray')
    plt.colorbar()
    plt.title('Mean Prediction')
    
    # Display the prediction uncertainty
    plt.subplot(2, 2, 4)
    plt.imshow(std_deviation[0, :, :, 0], cmap='gray')
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
    
    # Display the lower bound of the confidence interval
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(lower_bound[0, :, :, 0], cmap='gray')
    plt.colorbar()
    plt.title(f'{confidence_level * 100}% Confidence Lower Bound')
    
    # Display the upper bound of the confidence interval
    plt.subplot(1, 2, 2)
    plt.imshow(upper_bound[0, :, :, 0], cmap='gray')
    plt.colorbar()
    plt.title(f'{confidence_level * 100}% Confidence Upper Bound')
    
    plt.show()


def plot_correlation_analysis(mean_prediction, std_deviation):
    """
    Plot correlation between mean predictions and their standard deviations.
    
    Args:
    mean_prediction (numpy.ndarray): Mean predictions (height, width).
    std_deviation (numpy.ndarray): Standard deviations of predictions (height, width).
    """
    flat_mean = mean_prediction.flatten()
    flat_std = std_deviation.flatten()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(flat_mean, flat_std, alpha=0.1)
    plt.xlabel('Mean Prediction')
    plt.ylabel('Standard Deviation')
    plt.title('Correlation between Mean Prediction and Standard Deviation')
    plt.grid(True)
    plt.show()


def get_uncertainty_avgs(mean_prediction, std_deviation):
    flat_mean = mean_prediction.flatten()
    flat_std = std_deviation.flatten()
    
    # Segment the data based on mean prediction ranges
    low_uncertainty = flat_std[flat_mean < 0.2]
    medium_uncertainty = flat_std[(flat_mean >= 0.2) & (flat_mean < 0.8)]
    high_uncertainty = flat_std[flat_mean >= 0.8]

    # Calculate and print statistics for each segment
    print("Low Uncertainty Average:", np.mean(low_uncertainty))
    print("Medium Uncertainty Average:", np.mean(medium_uncertainty))
    print("High Uncertainty Average:", np.mean(high_uncertainty))

    logger.info(f'Low Uncertainty Average: {np.mean(low_uncertainty)}')
    logger.info(f'Medium Uncertainty Average: {np.mean(medium_uncertainty)}')
    logger.info(f'High Uncertainty Average: {np.mean(high_uncertainty)}')

def plot_hexbin(mean_prediction, std_deviation):
    flat_mean = mean_prediction.flatten()
    flat_std = std_deviation.flatten()
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(flat_mean, flat_std, gridsize=50, cmap='Blues', mincnt=1)
    cb = plt.colorbar(hb, label='Count')
    plt.xlabel('Mean Prediction')
    plt.ylabel('Standard Deviation')
    plt.title('Hexbin Plot of Mean Prediction vs. Standard Deviation')
    plt.grid(True)
    plt.show()

def plot_2d_histogram(mean_prediction, std_deviation):
    flat_mean = mean_prediction.flatten()
    flat_std = std_deviation.flatten()
    plt.figure(figsize=(8, 6))
    plt.hist2d(flat_mean, flat_std, bins=50, cmap='Blues')
    cb = plt.colorbar(label='Count')
    plt.xlabel('Mean Prediction')
    plt.ylabel('Standard Deviation')
    plt.title('2D Histogram of Mean Prediction vs. Standard Deviation')
    plt.grid(True)
    plt.show()