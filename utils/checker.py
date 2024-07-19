from utils.data_loader import (
    load_and_process_files,
    write_tfrecord,
    create_tf_dataset_from_tfrecord,
)
from configs import (
    TRAIN_IMAGES_DIR,
    TRAIN_MASKS_DIR,
    TEST_IMAGES_DIR,
    TEST_MASKS_DIR,
    PREPPED_TRAIN_DATASET,
    PREPPED_TEST_DATASET,
)
from pathlib import Path


def check_dirs():
    """Check if directories exist, if not create them"""
    directories = ["checkpoints", "data", "prepped_data", "model", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def check_prepped_data(get_train=True, get_test=True):
    """Check if prepped data exists, if not create it"""
    print("--" * 20)
    paths = {
        "train": (PREPPED_TRAIN_DATASET, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR),
        "test": (PREPPED_TEST_DATASET, TEST_IMAGES_DIR, TEST_MASKS_DIR),
    }

    dataset = dict()

    if get_train:
        prepped_data_path, images_dir, masks_dir = paths["train"]
        if Path(prepped_data_path).exists():
            print(f"{prepped_data_path} exists. Creating train dataset from it...")
            train_dataset = create_tf_dataset_from_tfrecord([str(prepped_data_path)])
        else:
            print(
                f"{prepped_data_path} does not exist. Processing train images and masks..."
            )
            images, masks = load_and_process_files(
                Path(images_dir), Path(masks_dir), prefix="train"
            )
            write_tfrecord(str(prepped_data_path), images, masks)
            train_dataset = create_tf_dataset_from_tfrecord([str(prepped_data_path)])

        print("--" * 20)
        dataset["train"] = train_dataset

    if get_test:
        prepped_data_path, images_dir, masks_dir = paths["test"]
        if Path(prepped_data_path).exists():
            print(f"{prepped_data_path} exists. Creating test dataset from it...")
            test_dataset = create_tf_dataset_from_tfrecord([str(prepped_data_path)])
        else:
            print(
                f"{prepped_data_path} does not exist. Processing test images and masks..."
            )
            images, masks = load_and_process_files(
                Path(images_dir), Path(masks_dir), prefix="test"
            )
            write_tfrecord(str(prepped_data_path), images, masks)
            test_dataset = create_tf_dataset_from_tfrecord([str(prepped_data_path)])

        print("--" * 20)
        dataset["test"] = test_dataset

    return dataset
