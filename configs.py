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

# Model hyperparameters
DROPOUT_RATE = 0.35
X_DIMENSION = 256
Y_DIMENSION = 256

# Uncertainty quantification test example index
UQ_TEST_EXAMPLE_INDEX = 0
NUM_SAMPLES_MC_DROPOUT_PREDICTION = 100