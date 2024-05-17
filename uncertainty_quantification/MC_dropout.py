import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.logger_prep import get_logger
from configs import X_DIMENSION, Y_DIMENSION
from utils.custom_funcs import dice_coefficient

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

    for image in tqdm(images, desc=f'MC Dropout Inference for all test images ({num_samples} times each)'):
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        mc_samples = [model(image, training=True) for _ in range(num_samples)]
        mc_samples = np.array(mc_samples)

        mean_prediction = np.mean(mc_samples, axis=0)
        std_deviation = np.std(mc_samples, axis=0)
        
        all_mean_predictions.append(mean_prediction)
        all_std_deviations.append(std_deviation)
    
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

    dice = dice_coefficient(test_mask, mean_prediction)

    plt.figure(figsize=(10, 10))
    

    plt.subplot(2, 2, 1)
    plt.imshow(test_image[:, :, :])
    plt.colorbar()
    plt.title('Test Image')
    
    plt.subplot(2, 2, 2)
    plt.imshow(test_mask[:, :, 0], cmap='gray')
    plt.colorbar()
    plt.title('Test Mask')
    
    plt.subplot(2, 2, 3)
    plt.imshow(mean_prediction[0, :, :, 0], cmap='gray')
    plt.colorbar()
    plt.title('Mean Prediction')
    
    plt.subplot(2, 2, 4)
    plt.imshow(std_deviation[0, :, :, 0], cmap='gray')
    plt.colorbar()
    plt.title('Prediction Uncertainty (std)')

    plt.suptitle(f'Dice Coefficient: {dice:.4f}', fontsize=16)
    
    plt.show()

def visualize_mean_std_grid(test_images, test_masks, predictions, rows=4, cols=5):
    """
    Visualize the test images, masks, mean predictions, and uncertainty across multiple samples.
    
    Parameters:
        test_images (numpy.array): Array of test images.
        test_masks (numpy.array): Array of corresponding masks.
        predictions (list): List of predictions arrays, each from a different MC sample.
        num_samples (int): Number of MC samples to visualize.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols + 10, 3 * rows + 10))
    
    for i in range(cols):
        for j in range(rows):
            ax = axes[j, i]
            mean_prediction = np.mean(predictions[i], axis=0)
            std_deviation = np.std(predictions[i], axis=0)
            
            if j == 0:  # First row - Test Image
                im = ax.imshow(test_images[i])
                ax.set_title(f'Test Image[{i+1}]')
            elif j == 1:  # Second row - Test Mask
                im = ax.imshow(test_masks[i][:, :, 0], cmap='gray')
                ax.set_title(f'Test Mask[{i+1}]')
            elif j == 2:  # Third row - Mean Prediction
                im = ax.imshow(mean_prediction[0, :, :, 0], cmap='gray')
                ax.set_title(f'Mean Prediction[{i+1}]')
            elif j == 3:  # Fourth row - Prediction Uncertainty
                im = ax.imshow(std_deviation[0, :, :, 0], cmap='gray')
                ax.set_title(f'Uncertainty[{i+1}]')
            
            ax.axis('off')
            if i == cols - 1:
                fig.colorbar(im, ax=ax)

    plt.tight_layout(pad=3.0, h_pad=3.0)
    plt.show()


def visualize_mean_std_grid_multi_models(test_images, test_masks, predictions, rows=4, cols=5, titles=None):
    """
    Visualize the test images, masks, mean predictions, and uncertainty across multiple samples.
    
    Parameters:
        test_images (numpy.array): Array of test images.
        test_masks (numpy.array): Array of corresponding masks.
        predictions (list): List of predictions arrays, each from a different MC sample.
        num_samples (int): Number of MC samples to visualize.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols + 10, 3 * rows + 10))
    
    for i in range(cols):
        for j in range(rows):
            ax = axes[j, i]
            mean_prediction = np.mean(predictions[i], axis=0)
            std_deviation = np.std(predictions[i], axis=0)
            
            if j == 0:  # First row - Test Image
                im = ax.imshow(test_images)
                ax.set_title(f'{titles[i]}')
            elif j == 1:  # Second row - Test Mask
                im = ax.imshow(test_masks, cmap='gray')
                ax.set_title(f'Test Mask')
            elif j == 2:  # Third row - Mean Prediction
                im = ax.imshow(mean_prediction[0, :, :, 0], cmap='gray')
                ax.set_title(f'Mean Prediction')
            elif j == 3:  # Fourth row - Prediction Uncertainty
                im = ax.imshow(std_deviation[0, :, :, 0], cmap='gray')
                ax.set_title(f'Uncertainty')
            
            ax.axis('off')
            if i == cols - 1:
                fig.colorbar(im, ax=ax)

    plt.tight_layout(pad=3.0, h_pad=3.0)
    plt.show()


# Confidence Interval Visualizations
def visualize_confidence_intervals(test_image, mean_prediction, std_deviation, confidence_level=0.95):
    """ Visualize the confidence intervals of the model predictions.
        95% lower-bound means that 95% of the predictions will be above this value.
        95% upper-bound means that 95% of the predictions will be below this value.
    """
    lower_bound = mean_prediction - std_deviation * 1.96
    upper_bound = mean_prediction + std_deviation * 1.96
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(lower_bound[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f'{confidence_level * 100}% Confidence Lower Bound')
    
    plt.subplot(1, 2, 2)
    plt.imshow(upper_bound[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
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
    
    low_uncertainty = flat_std[flat_mean < 0.2]
    medium_uncertainty = flat_std[(flat_mean >= 0.2) & (flat_mean < 0.8)]
    high_uncertainty = flat_std[flat_mean >= 0.8]
    
    print("Low Uncertainty Average:", np.mean(low_uncertainty))
    print("Medium Uncertainty Average:", np.mean(medium_uncertainty))
    print("High Uncertainty Average:", np.mean(high_uncertainty))

    logger.info(f'Low Uncertainty Average: {np.mean(low_uncertainty)}')
    logger.info(f'Medium Uncertainty Average: {np.mean(medium_uncertainty)}')
    logger.info(f'High Uncertainty Average: {np.mean(high_uncertainty)}')