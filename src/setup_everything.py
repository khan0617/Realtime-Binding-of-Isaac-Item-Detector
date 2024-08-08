"""
setup_everything.py does the following:
1. Download and scrape info/images for all IsaacItems form the Isaac Wiki
    a. Dump the IsaacItem data to: dumped_isaac_items.json
2. Generate a set of augmented IsaacItem images (rotated, flipped, etc.)
    a. These are stored in data/items/item_id/...augmented images here...
3. Overlay the augmented images onto backgrounds, stored in data/overlays/
4. Re-format all the data to use it with the YOLO model
    a. This empties out data/overlays and creates yolo_isaac_dataset
"""

import logging
import os
import sys

from constants import (
    BACKGROUND_DIR,
    DATA_DIR,
    ITEM_DIR,
    NUM_IMAGES_TO_USE_DURING_OVERLAY,
    OVERALL_DATASET_ROOT,
    OVERLAY_DIR,
    OVERLAYABLE_AREA,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VALID_RATIO,
    YOLO_DATASET_IMAGE_DIR,
    YOLO_DATASET_ROOT,
    YOLO_DATASET_TEST_DIR,
    YOLO_DATASET_TRAIN_DIR,
    YOLO_DATASET_VALID_DIR,
)
from generate_augmented_items import main as generate_augmented_items_main
from image_processing.image_overlay_processor import ImageOverlayProcessor
from isaac_yolo import isaac_yolo_dataset_generator
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    # see each respective module's main() for much more information on what the functions do and how to use them.
    logger.info("main: Step 1: Clean data/items")
    original_argv = sys.argv
    sys.argv = ["generated_augmented_items.py", "--clean"]
    generate_augmented_items_main()

    logger.info("main: Step 2: Download Isaac images from wiki (if needed) then augment images")
    sys.argv = ["generated_augmented_items.py", "--no-confirm"]
    generate_augmented_items_main()
    sys.argv = original_argv

    logger.info("main: Step 3: Overlay the images onto backgrounds")
    processor = ImageOverlayProcessor(
        data_dir=DATA_DIR, background_dir=BACKGROUND_DIR, item_dir=ITEM_DIR, output_dir=OVERLAY_DIR
    )
    processor.overlay_items_on_backgrounds(
        overlay_area=OVERLAYABLE_AREA, num_images_to_use=NUM_IMAGES_TO_USE_DURING_OVERLAY, seed=SEED
    )
    processor.plot_random_overlays_with_bboxes(num_images=3)

    logger.info("main: Step 4: Generate the YOLO dataset")
    isaac_yolo_dataset_generator.create_dataset_directories(root_dir=YOLO_DATASET_ROOT)
    pairs = isaac_yolo_dataset_generator.get_image_label_file_pairs(overlays_dir=os.path.join(DATA_DIR, OVERLAY_DIR))
    train_pairs, valid_pairs, test_pairs = isaac_yolo_dataset_generator.split_dataset(
        pairs=pairs, train_ratio=TRAIN_RATIO, valid_ratio=VALID_RATIO, test_ratio=TEST_RATIO
    )

    dir_to_copy_to = os.path.join(OVERALL_DATASET_ROOT, YOLO_DATASET_ROOT)
    isaac_yolo_dataset_generator.copy_files_to_yolo_dataset(train_pairs, YOLO_DATASET_TRAIN_DIR, dir_to_copy_to)
    isaac_yolo_dataset_generator.copy_files_to_yolo_dataset(valid_pairs, YOLO_DATASET_VALID_DIR, dir_to_copy_to)
    isaac_yolo_dataset_generator.copy_files_to_yolo_dataset(test_pairs, YOLO_DATASET_TEST_DIR, dir_to_copy_to)
    isaac_yolo_dataset_generator.delete_overlays_dir(os.path.join(DATA_DIR, OVERLAY_DIR))
    isaac_yolo_dataset_generator.generate_yolo_yaml_config(
        root_dir=YOLO_DATASET_ROOT,
        image_dir=YOLO_DATASET_IMAGE_DIR,
        train_dir=YOLO_DATASET_TRAIN_DIR,
        valid_dir=YOLO_DATASET_VALID_DIR,
        test_dir=YOLO_DATASET_TEST_DIR,
    )
    logger.info("main: All done! %s is good to go", YOLO_DATASET_ROOT)


if __name__ == "__main__":
    main()
