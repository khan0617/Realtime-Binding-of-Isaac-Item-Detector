"""
`isaac_yolo_dataset_generator.py`

Takes data/overlay/ and creates a dataset usable by YOLO, including train/test/valid directories.
"""

import logging
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import yaml
from tqdm import tqdm

from constants import (
    DATA_DIR,
    OVERALL_DATASET_ROOT,
    OVERLAY_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VALID_RATIO,
    YOLO_DATASET_IMAGE_DIR,
    YOLO_DATASET_LABEL_DIR,
    YOLO_DATASET_ROOT,
    YOLO_DATASET_TEST_DIR,
    YOLO_DATASET_TRAIN_DIR,
    YOLO_DATASET_VALID_DIR,
    YOLO_DATASET_YAML_FILE,
)
from logging_config import configure_logging
from utils import get_id_name_mapping

configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class ImageLabelPair:
    """Represents an image and its associated YOLO label file.

    They will be the same except for the extension.
    Ex: image_path=".../100_rotate_1234.jpg", label_path=".../100_rotate_1234.txt"
    """

    image_path: str
    label_path: str


def create_dataset_directories(root_dir: str) -> None:
    """
    Create the directory structure for the YOLO dataset.

    If root_dir == "yolo_isaac_dataset" then the final directory created will be like:
    .
    └── datasets/
        └── yolo_isaac_dataset/
            ├── images/
            │   ├── train/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   ├── val/
            │   │   ├── img2.jpg
            │   │   └── ...
            │   └── test/
            │       ├── img3.jpg
            │       └── ..
            └── labels/
                ├── train/
                │   ├── img1.txt
                │   └── ...
                ├── val/
                │   ├── img2.txt
                │   └── ...
                └── test/
                    ├── img3.txt
                    └── ...
    Args:
        root_dir (str): The root of the YOLO dataset (ex. "yolo_isaac_dataset")
    """
    root_dir = os.path.join(OVERALL_DATASET_ROOT, root_dir)
    logger.info("create_dataset_directories: Creating all directories for YOLO dataset.")
    for subdir in [YOLO_DATASET_IMAGE_DIR, YOLO_DATASET_LABEL_DIR]:
        full_subdir_path = os.path.join(root_dir, subdir)
        os.makedirs(full_subdir_path, exist_ok=True)
        for split in [YOLO_DATASET_TRAIN_DIR, YOLO_DATASET_VALID_DIR, YOLO_DATASET_TEST_DIR]:
            full_split_path = os.path.join(full_subdir_path, split)
            os.makedirs(full_split_path, exist_ok=True)
    logger.info("create_dataset_directories: Done! Created all necessary directories.")


def get_image_label_file_pairs(overlays_dir: str) -> list[ImageLabelPair]:
    """Get the list of all pairs of (image.jpg, image_.txt) within overlays_dir,
    where image.txt represents the YOLO bbox file.

    overlays_dir should be formatted like:
    .
    └── overlays
        ├── background_1
        │   ├── image1.jpg
        │   ├── image1.txt
        │   └── ...
        ├── background_2
        │   ├── image2.jpg
        │   ├── image2.txt
        │   └── ...
        └── ...
    This function assumes that every "image.jpg" has an associated "image.txt".

    Args:
        overlays_dir (str): The root directory of all images to find image/label pairs.
            Ex: data/overlays

    Returns:
        list[ImageLabelPair]: The list of all (image_path, label_path) filename tuples.
    """
    logger.info("get_image_label_file_pairs: Generating pairs from %s...", overlays_dir)
    image_label_pairs: list[ImageLabelPair] = []
    for root, _, files in os.walk(overlays_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg")):
                image_path = os.path.join(root, file)
                label_path = f"{os.path.splitext(image_path)[0]}.txt"

                if not os.path.exists(label_path):
                    msg = f"image path {image_path} exists but label path {label_path} does not."
                    logger.error(msg)
                    raise ValueError(msg)

                image_label_pairs.append(ImageLabelPair(image_path, label_path))

    logger.info("get_image_label_file_pairs: Done! Generated %d pairs", len(image_label_pairs))
    return image_label_pairs


def split_dataset(
    pairs: list[ImageLabelPair], train_ratio: float, valid_ratio: float, test_ratio: float, seed: int | None = None
) -> tuple[list[ImageLabelPair], list[ImageLabelPair], list[ImageLabelPair]]:
    """
    Split the dataset into training, validation, and test sets.

    Args:
        pairs (List[ImageLabelPair]): The list of all image/label filename pairs.
        train_ratio (float): The ratio of the training set.
        valid_ratio (float): The ratio of the validation set.
        test_ratio (float): The ratio of the test set.
        seed (int, optional): Seed for random shuffling of the pairs.

    Returns:
        tuple: Three lists containing the training, validation, and test pairs respectively.
    """
    logger.info("split_dataset: Splitting dataset into train/valid/test...")
    assert train_ratio + valid_ratio + test_ratio == 1, "train, valid, and test ratios must sum to 1"
    random.seed(seed)
    random.shuffle(pairs)
    num_total = len(pairs)
    num_train = int(num_total * train_ratio)
    num_valid = int(num_total * valid_ratio)
    num_test = max(int(num_total * test_ratio), num_total - num_train - num_valid)  # don't miss any due to rounding
    assert num_train + num_valid + num_test == num_total, "num_train + num_valid + num_test must equal num_total"

    logger.info(
        "split_dataset: train/valid/test have respective sizes: %d, %d, %d, total: %d",
        num_train,
        num_valid,
        num_test,
        num_total,
    )

    train_pairs = pairs[:num_train]
    valid_pairs = pairs[num_train : num_train + num_valid]
    test_pairs = pairs[num_train + num_valid : num_train + num_valid + num_test]

    # fmt: off
    assert len(train_pairs) + len(valid_pairs) + len(test_pairs) == len(pairs),\
        "lengths of train, valid, and test pairs must sum to len(pairs)."
    # fmt: on

    logger.info(
        "split_dataset: Done! Training set: %d, Validation set: %d, Test set: %d",
        len(train_pairs),
        len(valid_pairs),
        len(test_pairs),
    )

    return train_pairs, valid_pairs, test_pairs


def copy_files_to_yolo_dataset(pairs: list[ImageLabelPair], split_name: str, root_dir: str) -> None:
    """
    Copy image and label files to their respective directories in the YOLO dataset.

    Args:
        pairs (list[ImageLabelPair]): The list of image/label pairs to copy.
        split_name (str): The name of the dataset split ('train', 'valid', or 'test').
        root_dir (str): The root directory of the YOLO dataset, ex: "datasets/yolo_isaac_dataset"
    """
    full_image_dir = os.path.join(root_dir, YOLO_DATASET_IMAGE_DIR, split_name)
    full_label_dir = os.path.join(root_dir, YOLO_DATASET_LABEL_DIR, split_name)
    logger.info("copy_files_to_yolo_dataset: Copying files to %s...", split_name)

    def copy_helper(pair: ImageLabelPair) -> None:
        try:
            shutil.copy(pair.image_path, full_image_dir)
            shutil.copy(pair.label_path, full_label_dir)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Failed to copy either %s or %s to %s or %s: %s!",
                pair.image_path,
                pair.label_path,
                full_image_dir,
                full_label_dir,
                str(e),
            )
            sys.exit(1)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(copy_helper, pairs), desc="Copying images/labels (multi-threaded)", total=len(pairs)))

    logger.info("copy_files_to_yolo_dataset: Done! Copied %d pairs to %s", len(pairs), split_name)


def delete_overlays_dir(overlays_dir: str) -> None:
    """Delete overlays_dir.

    Irreversible, make sure you have called copy_files_to_yolo_dataset(...) beforehand!

    Args:
        overlays_dir (str): Full path to the overlays directory. Ex: data/overlays
    """
    logger.info("delete_overlays_dir: Deleting %s...", overlays_dir)
    try:
        shutil.rmtree(overlays_dir)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "delete_overlays_dir: Couldn't delete %s (maybe you already deleted it?): %s", overlays_dir, str(e)
        )
        sys.exit(1)
    logger.info("delete_overlays_dir: Done! Deleted %s.", overlays_dir)


def generate_yolo_yaml_config(
    root_dir: str,
    image_dir: str,
    train_dir: str,
    valid_dir: str,
    test_dir: str,
    yaml_filename: str = YOLO_DATASET_YAML_FILE,
) -> None:
    """
    Generate a YOLO configuration file in YAML format.
    The "data.yaml" file will be created in the directory this script is executed.
    Basically, be sure to run this script from realtime-binding-of-isaac-item-detector/
    So that "data.yaml" ends up in the root of the project.

    See: https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format

    Args:
        root_dir (str): The root directory of the YOLO dataset. Ex. "yolo_isaac_dataset"
        image_dir: (str): The name of the directory holding images in root_dir. Ex: "images"
        train_dir (str): The directory containing training images. Ex: "train"
        valid_dir (str): The directory containing validation images. Ex: "val"
        test_dir (str): The directory containing test images. Ex: "test"
        yaml_filename (str): Name of the generated yaml file. Ex: "data.yaml"
    """
    name_id_map = get_id_name_mapping()
    config = {
        "path": root_dir,
        "train": os.path.join(image_dir, train_dir),
        "val": os.path.join(image_dir, valid_dir),
        "test": os.path.join(image_dir, test_dir),
        "names": name_id_map,
    }

    with open(yaml_filename, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    logger.info("generate_yolo_yaml_config: Generated YOLO config file: %s", yaml_filename)


def main() -> None:  # pylint: disable=missing-function-docstring
    # example usage: 1st create the necessary directories
    create_dataset_directories(root_dir=YOLO_DATASET_ROOT)

    # collect all image/label pairs then split them into train/valid/test
    pairs = get_image_label_file_pairs(overlays_dir=os.path.join(DATA_DIR, OVERLAY_DIR))
    train_pairs, valid_pairs, test_pairs = split_dataset(
        pairs=pairs, train_ratio=TRAIN_RATIO, valid_ratio=VALID_RATIO, test_ratio=TEST_RATIO
    )

    # move files to their new respective directories
    dir_to_copy_to = os.path.join(OVERALL_DATASET_ROOT, YOLO_DATASET_ROOT)
    copy_files_to_yolo_dataset(train_pairs, YOLO_DATASET_TRAIN_DIR, dir_to_copy_to)
    copy_files_to_yolo_dataset(valid_pairs, YOLO_DATASET_VALID_DIR, dir_to_copy_to)
    copy_files_to_yolo_dataset(test_pairs, YOLO_DATASET_TEST_DIR, dir_to_copy_to)

    # optional, to save space we can delete the original dataset
    delete_overlays_dir(os.path.join(DATA_DIR, OVERLAY_DIR))

    # finally, generate the .yaml file which the yolo model needs
    generate_yolo_yaml_config(
        root_dir=YOLO_DATASET_ROOT,
        image_dir=YOLO_DATASET_IMAGE_DIR,
        train_dir=YOLO_DATASET_TRAIN_DIR,
        valid_dir=YOLO_DATASET_VALID_DIR,
        test_dir=YOLO_DATASET_TEST_DIR,
    )


if __name__ == "__main__":
    main()
