import numpy as np
from tqdm import tqdm

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


