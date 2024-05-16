# Data configuration
X_DIMENSION = 256
Y_DIMENSION = 256

# Directory paths
TRAIN_IMAGES_DIR = './data/images/train'
TRAIN_MASKS_DIR = './data/mask/train'
TEST_IMAGES_DIR = './data/images/val'
TEST_MASKS_DIR = './data/mask/val'

# Prepped directories, images and masks resized to 256x256 and made into numpy arrays and grayscale
PREPPED_TRAIN_IMAGES = 'prepped_data/trainimages.npy'
PREPPED_TRAIN_MASKS = 'prepped_data/trainmasks.npy'
PREPPED_TEST_IMAGES = 'prepped_data/testimages.npy'
PREPPED_TEST_MASKS = 'prepped_data/testmasks.npy'

# Model paramaeters/hyperparameters
DROPOUT_RATE = 0.35
BATCH_SIZE = 1
EPOCHS = 5

# Pick from : 
# 'relu'      : ReLU is the most commonly used activation function in deep learning. It helps in mitigating the vanishing gradient problem and is computationally efficient
# 'elu'       : ELU can result in faster and more accurate learning by having negative values which push mean activations closer to zero.
# 'swish'     : Swish tends to outperform ReLU in deeper networks by avoiding the dead neuron problem and providing a smooth activation function.
# 'gelu'      : GELU smooths the output curve and tends to perform better on tasks involving natural language processing, and it may offer benefits in image tasks as well.
# 'leaky_relu': Leaky ReLU allows a small gradient when the unit is not active, which helps in avoiding dead neurons.
ACTIVATION_FUNC = 'relu'


# Uncertainty quantification testing
NUM_SAMPLES_MC_DROPOUT_PREDICTION = 100
GRID_ITERATIONS = 5