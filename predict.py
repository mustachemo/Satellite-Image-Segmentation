from tensorflow.keras.models import load_model
import logging
import matplotlib.pyplot as plt
from utils.data_loader_unet import test_images, test_masks

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load the model
    try: 
        model = load_model('checkpoints/unet_model.h5')
        logging.info('Model loaded successfully')
    except Exception as e:
        logging.error(f'Error: {e}')
        exit()

    # Predict and show results
    logging.info('Predicting test images')
    predictions = model.predict(test_images)
    for i in range(5):
        # Display the image and mask for prediction
        fig, ax = plt.subplots(1, 3, figsize=(15, 7))

        ax[0].imshow(test_images[i])
        ax[0].set_title('Image')

        ax[1].imshow(test_masks[i], cmap='gray')
        ax[1].set_title('Mask')

        ax[2].imshow(predictions[i], cmap='gray')
        ax[2].set_title('Predicted Mask')

        plt.show()