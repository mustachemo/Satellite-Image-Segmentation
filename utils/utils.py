from configs import BATCH_SIZE, MIXED_PRECISION


def calculate_memory_usage(image, mask):
    bytes = 2 if MIXED_PRECISION else 4
    image_size = image.shape[0] * image.shape[1] * image.shape[2] * bytes
    mask_size = mask.shape[0] * mask.shape[1] * mask.shape[2] * bytes
    batch_size = BATCH_SIZE * (image_size + mask_size)
    return batch_size / 1024**2
