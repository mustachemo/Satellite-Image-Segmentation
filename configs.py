# Directory paths
train_images_dir = './data/images/train'
train_masks_dir = './data/mask/train'
test_images_dir = './data/images/val'
test_masks_dir = './data/mask/val'

# Prepped directories, images and masks resized to 256x256 and made into numpy arrays and grayscale
prepped_train_images = 'prepped_data/trainimages.npy'
prepped_train_masks = 'prepped_data/trainmasks.npy'
prepped_test_images = 'prepped_data/testimages.npy'
prepped_test_masks = 'prepped_data/testmasks.npy'