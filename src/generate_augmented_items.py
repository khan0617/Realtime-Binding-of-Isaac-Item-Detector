import argparse
import logging
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import chain, combinations, repeat

from tqdm import tqdm

from constants import DATA_DIR as _DATA_DIR  # I redefine a DEFAULT_DATA_DIR here and don't want to mix them up
from constants import ITEM_DIR, JSON_DUMP_FILE
from constants import SEED as _SEED
from constants import UNMODIFIED_FILE_NAME, WIKI_ITEMS_HOMEPAGE
from image_processing import data_augmentor
from image_processing.augmentation import Augmentation
from logging_config import configure_logging
from scraping import scraper
from scraping.isaac_item import IsaacItem

configure_logging()
logger = logging.getLogger(__name__)

AUGMENTATIONS_TO_APPLY = [
    Augmentation.ROTATE,
    Augmentation.NOISE,
    Augmentation.COLOR_JITTER,
    Augmentation.VERTICAL_FLIP,
    Augmentation.HORIZONTAL_MIRROR,
    Augmentation.BRIGHTNESS,
    Augmentation.CONTRAST,
    Augmentation.TRANSLATE,
    Augmentation.SHEAR,
    Augmentation.SHARPNESS,
]

DEFAULT_NUM_AUGMENTED = 2  # how many images to generate per combination of augmentation
DEFAULT_SEED = _SEED
DEFAULT_MAX_SUBSET_SIZE = 2
DEFAULT_DATA_DIR = _DATA_DIR
DEFAULT_ITEM_DIR = ITEM_DIR


class _TypedArgparseNamespace(argparse.Namespace):
    """Typed namespace to add code completion to the output of parser.parse_args()."""

    num_augmented: int
    seed: int
    max_subset_size: int
    data_dir: str
    item_dir: str
    clean: bool
    no_confirm: bool


def _get_non_empty_subsets(iterable: Iterable, max_subset_size: int | None = None) -> chain:
    """
    Generate non-empty subsets of the iterable up to the specified maximum size.

    See https://stackoverflow.com/questions/27974126/get-all-n-choose-k-combinations-of-length-n.

    Args:
        iterable (iterable): The iterable from which to generate subsets.
        max_size (int, optional): The maximum size of subsets to include.
                                  If None, includes subsets up to the size of the iterable.

    Returns:
        chain: A chain over the non-empty subsets of the iterable.
    """
    s = list(iterable)
    if max_subset_size is None:
        max_subset_size = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(1, max_subset_size + 1))


def _augment_item_image(
    item: IsaacItem, aug_subsets: list[tuple[Augmentation]], num_augmented: int, full_item_dir: str, seed: int
) -> None:
    """
    Augment images for a single Isaac item using specified augmentation combinations.

    This is mainly used as a helper for ProcessPoolExecutor for parallelism.

    Args:
        item (IsaacItem): The IsaacItem object containing details about the item.
        aug_subsets (list[tuple[Augmentation]]): List of augmentation combinations to apply.
        num_augmented (int): Number of augmented images to generate per combination.
        full_item_dir (str): The directory containing folders for each item.
            I.e. if full_item_dir == "data/items/", then we should have data/items/145/, data/items/72/, etc.
        seed (int): A seed to observe reproducible results when running augment_image() repeatedly.
    """
    output_dir = os.path.join(full_item_dir, item.get_item_id_tail())
    image_path = os.path.join(output_dir, UNMODIFIED_FILE_NAME)
    if not os.path.exists(image_path):
        logger.warning("Image not found for item: %s, expected at: %s", item.name, image_path)
        return

    data_augmentor.augment_image(
        image_path=image_path,
        output_dir=output_dir,
        augmentations=aug_subsets,
        num_augmented=num_augmented,
        seed=seed,
    )


def _delete_augmented_images_from_item_dir(item_dir: str) -> None:
    """Delete all augmented images in the given directory, keeping only the original image.

    This is a helper for use with ThreadPoolExecutor.
    This function does not traverse any subdirectories if any exist.

    Args:
        item_dir (str): The directory which contains all images for a single item.
    """
    for file in os.listdir(item_dir):
        if file != UNMODIFIED_FILE_NAME:
            file_path = os.path.join(item_dir, file)
            try:
                os.remove(file_path)
                logger.debug("_delete_augmented_images_from_item_dir: Removed %s", file_path)
            except OSError as e:
                logger.error("_delete_augmented_images_from_item_dir: Failed to remove %s: %s", file_path, str(e))


def _clean_data_dir(full_item_dir: str) -> None:
    """Remove any augmented images from the specified data_dir.

    Args:
        full_item_dir (str): The items/ directory (like data/items/). Within item_dir there should be subdirectories for each item.
    """
    # fmt: off
    logger.info("clean_data_dir: Cleaning data directory: %s/ ...", full_item_dir)
    subdirs = [
        os.path.join(full_item_dir, d)
        for d in os.listdir(full_item_dir)
        if os.path.isdir(os.path.join(full_item_dir, d))
    ]
    # fmt: on

    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(_delete_augmented_images_from_item_dir, subdirs), desc="Removing files", total=len(subdirs)
            )
        )
    logger.info("clean_data_dir: Done! Removed augmented images from: %s/", full_item_dir)


def main() -> None:
    """
    Generate or clean a dataset of augmented images for Isaac items.

    This script performs several things depending on the specified command-line arguments:
    1. If the `--clean` flag is used, it removes all augmented images from the specified data directory,
       keeping only the original images. Using --clean ignores all other arguments, so don't use --clean and
       specify other arguments.
    2. If the `--clean` flag is not used, the script will:
       - Fetch the latest data from the Isaac item wiki and parse the item details.
       - Download images for each item into the specified data directory.
       - Generate a specified number of augmented images per item using combinations of augmentations.

    Augmentations are stored in the same directory as the original item images, with filenames indicating
    the augmentations performed.

    Command-line Arguments (all optional, defaults are provided, see --help):
    - `--help`: Display information about this program and its arguments.
    - `--num_augmented`: Number of augmented images to generate per augmentation combination.
    - `--seed`: Random seed for reproducibility.
    - `--max_subset_size`: Maximum number of augmentations to apply at once to an image.
    - `--data_dir`: Root directory for the data.
    - `--item_dir`: Subdirectory in --data_dir to store augmented images.
    - `--clean`: If set, only cleans the data directory of augmented images.
    - `--no_confirm` If set, skip the confirmation step when using all defaults.

    Example usage:
        \n`python generate_augmented_items.py --num_augmented 5 --seed 123 --max_subset_size 3 --data_dir my_data`
        \n`python generate_augmented_items.py --no_confirm` (use all defaults and skip the confirmation step)
    """
    # fmt: off
    parser = argparse.ArgumentParser(
        description=(
            "Generate or clean your dataset of augmented images for Isaac items. "
            "If --clean is specified along with other commands, only clean will execute; "
            "other commands will be ignored."
        )
    )
    parser.add_argument("--num_augmented", type=int, default=DEFAULT_NUM_AUGMENTED, help=f"Number of augmented images per combo (default: {DEFAULT_NUM_AUGMENTED}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).")
    parser.add_argument("--max_subset_size", type=int, default=DEFAULT_MAX_SUBSET_SIZE, help=f"Max size of augmentation subsets (default: {DEFAULT_MAX_SUBSET_SIZE}).")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help=f"Root directory for the data. (default: '{DEFAULT_DATA_DIR}').")
    parser.add_argument("--item_dir", type=str, default=DEFAULT_ITEM_DIR, help=f"Subdirectory in --data_dir to store augmented images (default: '{DEFAULT_ITEM_DIR}').")
    parser.add_argument("--clean", action="store_true", help="Clean the data directory of augmented images.")
    parser.add_argument("--no-confirm", action="store_true", help="Skip the confirmation prompt when running with default settings.")
    # fmt: on

    args: _TypedArgparseNamespace = parser.parse_args()  # type: ignore
    all_defaults_used = (
        args.num_augmented == DEFAULT_NUM_AUGMENTED
        and args.seed == DEFAULT_SEED
        and args.max_subset_size == DEFAULT_MAX_SUBSET_SIZE
        and args.data_dir == DEFAULT_DATA_DIR
        and args.item_dir == DEFAULT_ITEM_DIR
    )

    full_item_dir = os.path.join(args.data_dir, args.item_dir)

    if args.clean:
        _clean_data_dir(full_item_dir)
        return

    if all_defaults_used and not args.no_confirm:
        confirm = input("No options specified. Proceed with default settings? (y/n): ").strip()
        if confirm.lower() not in ["y", "yes"]:
            print("Operation cancelled.")
            return

    # get the isaac items from the html response, then dump to json.
    html = scraper.fetch_page(WIKI_ITEMS_HOMEPAGE)
    isaac_items = scraper.parse_isaac_items_from_html(html)
    scraper.download_item_images(isaac_items, args.data_dir, args.item_dir)
    scraper.dump_item_data_to_json(isaac_items, JSON_DUMP_FILE)

    # note: if len(AUGMENTATIONS_TO_APPLY) == 10, we get 55 subsets when max_subset_size == 2.
    aug_subsets: list[tuple[Augmentation]] = list(_get_non_empty_subsets(AUGMENTATIONS_TO_APPLY, args.max_subset_size))

    # generate all the data augmentations in parallel
    logger.info("main: Generating augmentations, this may take a while...")
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    _augment_item_image,
                    isaac_items,
                    repeat(aug_subsets),
                    repeat(args.num_augmented),
                    repeat(full_item_dir),
                    repeat(args.seed),
                ),
                desc="Augmenting (Multi-processing)",
                total=len(isaac_items),
            )
        )
    logger.info("main: Done augmenting images!")


if __name__ == "__main__":
    main()
