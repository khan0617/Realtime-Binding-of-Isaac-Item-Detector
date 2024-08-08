from image_processing.bbox import CocoBbox

WIKI_HOMEPAGE_ROOT = "https://bindingofisaacrebirth.fandom.com/"
WIKI_ITEMS_HOMEPAGE = "https://bindingofisaacrebirth.fandom.com/wiki/Items"
CACHE_DIR = "scraper_cache"
CACHE_FILE = "cache.json"
DATA_DIR = "data"
ITEM_DIR = "items"
BACKGROUND_DIR = "isaac_backgrounds"
OVERLAY_DIR = "overlays"  # where all the item images (augmented + original) will be overlaid onto each background.
UNMODIFIED_FILE_NAME = "original_img.png"
JSON_DUMP_FILE = "dumped_isaac_items.json"
BROKEN_SHOVEL_ACTIVE_ID = "5.100.550"
BROKEN_SHOVEL_PASSIVE_ID = "5.100.551"
LOG_DIR = "logs"
LOG_FILENAME = "app.log"
TARGET_BACKGROUND_SIZE = (1000, 625)
ISAAC_ITEM_SCALE_FACTOR = 2.5
SEED = 39  # おかしいミク番号w
NUM_IMAGES_TO_USE_DURING_OVERLAY = 4

# the isaac item images should only be randomly placed with the area this box represents
# I used ImageOverlayProcessor.visualize_overlayable_area(...) with various combos to find this
OVERLAYABLE_AREA = CocoBbox(115, 105, 770, 415)

OVERALL_DATASET_ROOT = "datasets"  # required for ultralytics yolo file hierarchy
YOLO_DATASET_ROOT = "yolo_isaac_dataset"
YOLO_DATASET_IMAGE_DIR = "images"
YOLO_DATASET_LABEL_DIR = "labels"
YOLO_DATASET_TRAIN_DIR = "train"
YOLO_DATASET_VALID_DIR = "val"
YOLO_DATASET_TEST_DIR = "test"
YOLO_DATASET_YAML_FILE = "data.yaml"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
