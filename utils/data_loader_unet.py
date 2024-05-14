import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from functools import lru_cache, partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        # Adjust to TensorFlow; assumes saved as .npy or similar
        return Image.fromarray(np.load(filename))
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_files = list(mask_dir.glob(idx + mask_suffix + '.*'))
    if not mask_files:
        logging.error(f"No mask files found for index {idx} with suffix {mask_suffix}")
        raise FileNotFoundError(f"No mask files found for index {idx} with suffix {mask_suffix}")
    mask_file = mask_files[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class DatasetLoader(tf.data.Dataset):
    def _generator(ids, images_dir, mask_dir, scale, mask_suffix, mask_values):
        images_dir = Path(images_dir)
        mask_dir = Path(mask_dir)
        
        for id in ids:
            name = id
            mask_file = list(mask_dir.glob(name + mask_suffix + '.*'))
            img_file = list(images_dir.glob(name + '.*'))

            assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
            assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
            mask = load_image(mask_file[0])
            img = load_image(img_file[0])

            assert img.size == mask.size, \
                f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

            img = preprocess(mask_values, img, scale, is_mask=False)
            mask = preprocess(mask_values, mask, scale, is_mask=True)

            # yield {
            #     'image': tf.convert_to_tensor(img.copy(), dtype=tf.float32),
            #     'mask': tf.convert_to_tensor(mask.copy(), dtype=tf.int64)
            # }
            # yield {
            #     'input_1': tf.convert_to_tensor(img.copy(), dtype=tf.float32),
            #     'mask': tf.convert_to_tensor(mask.copy(), dtype=tf.int64)
            # }
            yield (tf.convert_to_tensor(img.copy(), dtype=tf.float32),
                tf.convert_to_tensor(mask.copy(), dtype=tf.int64))

    def __new__(cls, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '_mask'):
        images_dir = Path(images_dir)
        mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=mask_dir, mask_suffix=mask_suffix), ids),
                total=len(ids)
            ))

        mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {mask_values}')

        # return tf.data.Dataset.from_generator(
        #     cls._generator,
        #     output_types={'image': tf.float32, 'mask': tf.int64},
        #     args=(ids, str(images_dir), str(mask_dir), scale, mask_suffix, mask_values)
        # )
        # return tf.data.Dataset.from_generator(
        #     cls._generator,
        #     output_types={'input_1': tf.float32, 'mask': tf.int64},
        #     args=(ids, str(images_dir), str(mask_dir), scale, mask_suffix, mask_values)
        # )
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 256, 256), dtype=tf.int64)
            ),
            args=(ids, str(images_dir), str(mask_dir), scale, mask_suffix, mask_values)
        )


def preprocess(mask_values, pil_img, scale, is_mask):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img = np.asarray(pil_img)

    if is_mask:
        mask = np.zeros((newH, newW), dtype=np.int64)
        for i, v in enumerate(mask_values):
            if img.ndim == 2:
                mask[img == v] = i
            else:
                mask[(img == v).all(-1)] = i
        return mask
    else:
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))
        if (img > 1).any():
            img = img / 255.0
        return img
