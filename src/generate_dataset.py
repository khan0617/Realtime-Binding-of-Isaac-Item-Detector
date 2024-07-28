import logging
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, combinations, repeat

from constants import DATA_DIR, JSON_DUMP_FILE, UNMODIFIED_FILE_NAME, WIKI_ITEMS_HOMEPAGE
from data_augmentation.augmentation import Augmentation
from data_augmentation.data_augmentor import DataAugmentor
from logging_config import configure_logging
from scraping.isaac_item import IsaacItem
from scraping.scraper import Scraper

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

NUM_AUGMENTED = 2  # how many images to generate per combination of augmentation


def get_non_empty_subsets(iterable: Iterable, max_subset_size: int | None = None) -> chain:
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


def augment_item_image(item: IsaacItem, aug_subsets: list[tuple[Augmentation]], num_augmented: int) -> None:
    """
    Augment images for a single Isaac item using specified augmentation combinations.

    This is mainly used as a helper for ProcessPoolExecutor for parallelism.

    Args:
        item (IsaacItem): The IsaacItem object containing details about the item.
        aug_subsets (list[tuple[Augmentation]]): List of augmentation combinations to apply.
        num_augmented (int): Number of augmented images to generate per combination.
    """
    output_dir = os.path.join(DATA_DIR, item.img_dir)
    image_path = os.path.join(DATA_DIR, item.img_dir, UNMODIFIED_FILE_NAME)
    if not os.path.exists(image_path):
        logger.warning("Image not found for item: %s, expected at: %s", item.name, image_path)
        return

    DataAugmentor.augment_image(
        image_path=image_path,
        output_dir=output_dir,
        augmentations=aug_subsets,
        num_augmented=num_augmented,
        seed=39,  # ミクの番号w
    )


def main() -> None:
    """Populate the data directory with augmented images for each IsaacItem.

    All steps: fetch the Isaac item wiki's HTML, parse the IsaacItems from the html, dump the IsaacItems to JSON,
    download the original png for each item to {DATA_DIR}/{isaac_item.img_dir}/{UNMODIFIED_FILE_NAME},
    then apply a list of augmentations to each original image, generating NUM_AUGMENTED augmented images per augmentation combo.
    Augmentations are stored in the same directory as the original item image.
    """
    html = Scraper.fetch_page(WIKI_ITEMS_HOMEPAGE)
    isaac_items = Scraper.parse_isaac_items_from_html(html)
    Scraper.download_item_images(isaac_items, DATA_DIR)
    Scraper.dump_item_data_to_json(isaac_items, JSON_DUMP_FILE)

    # when len(AUGMENTATIONS_TO_APPLY) == 10, we get 55 subsets with this call.
    aug_subsets: list[tuple[Augmentation]] = list(get_non_empty_subsets(AUGMENTATIONS_TO_APPLY, 2))

    # generate all the data augmentations
    logger.info("main: Generating augmentations, this may take a while...")
    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(augment_item_image, isaac_items, repeat(aug_subsets), repeat(NUM_AUGMENTED))
    logger.info("main: Done augmenting images!")


if __name__ == "__main__":
    main()
