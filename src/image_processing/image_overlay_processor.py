import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageFile, ImageOps

from constants import (
    BACKGROUND_DIR,
    DATA_DIR,
    ISAAC_ITEM_SCALE_FACTOR,
    ITEM_DIR,
    OVERLAY_DIR,
    OVERLAYABLE_AREA,
    SEED,
    TARGET_BACKGROUND_SIZE,
)
from image_processing.bbox import CocoBbox, YoloBbox
from logging_config import configure_logging
from utils import read_yolo_label_file

configure_logging()
logger = logging.getLogger(__name__)


def _save_overlay_metadata(output_path: str, class_id: str, bbox: YoloBbox) -> None:
    """Save metadata for an item overlay as a .txt file in YOLO format.

    Each .txt label file consists of lines like: `<class_id> <x_center> <y_center> <width> <height>`
    If we had multiple items on screen we could supply multiple lines.
    We will use the Isaac item id as the yolo class ID.
    So for item_id "5.100.320", we'll use 320 (int) as the yolo class id.

    Example: 0 0.48 0.63 0.69 0.71

    Args:
        output_path (str): The path to save the .txt file (must end in .txt).
        class_id (str): The class_id we'll to represent this item to the YOLO model.
        bbox (YoloBbox): The bounding box of the overlaid item.
    """
    metadata = f"{class_id} {bbox.x_center} {bbox.y_center} {bbox.width} {bbox.height}"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(metadata)

    logger.debug("_save_overlay_metadata: Saved metadata at path %s", output_path)


def _overlay_augmented_images_on_background(
    overlay_area: CocoBbox,
    background: Image.Image,
    item_paths: list[str],
    item_id_tail: str,
    background_name: str,
    full_output_dir: str,
) -> None:
    """Overlay each image in item_paths onto background with random positioning within overlay_area.

    Args:
        overlay_area (CocoBbox): The bounding box area within which items can be overlaid.
        background (PIL.Image.Image): The background image onto which items will be overlaid.
        item_paths (List[str]): List of complete file paths to augmented item images.
        item_id_tail (str): Tail of the item id, such as "145" for Guppy's Head (which is item 5.100.145).
        background_name (str): The name of the background being used. (ex. "Library_7")
        full_output_dir (str): The output directory of these overlayed images (ex: "data/overlays")
    """
    for item_path in item_paths:
        # open the item image then resize it otherwise they're too small compared to in-game.
        item_image = Image.open(item_path).convert("RGBA")
        new_image_size = (
            int(item_image.width * ISAAC_ITEM_SCALE_FACTOR),
            int(item_image.height * ISAAC_ITEM_SCALE_FACTOR),
        )
        item_image = item_image.resize(new_image_size, Image.Resampling.LANCZOS)

        # get random paste location on the image
        x = random.randint(overlay_area.x, overlay_area.x + overlay_area.w - item_image.width)
        y = random.randint(overlay_area.y, overlay_area.y + overlay_area.h - item_image.height)
        item_bbox = CocoBbox(x, y, item_image.width, item_image.height)

        # overlay the item onto the background
        background_copy = background.copy()
        background_copy.paste(item_image, (x, y), item_image)  # PIL uses the alpha of the item image as a mask

        # generate the output filename
        output_filename = (
            f"{item_id_tail}_"  # ex. "145_"
            f"{os.path.splitext(os.path.basename(item_path))[0]}.jpg"  # ex. "rotate_flip_1234.jpg"
        )

        # these files will be saved into an output dir like "data/overlays/Library_7/
        output_dir = os.path.join(full_output_dir, background_name)
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(output_dir, exist_ok=True)

        # save the image and metadata
        background_copy.save(output_path)
        logger.debug("_overlay_augmented_images_on_background: Saved overlayed image: %s", output_path)
        _save_overlay_metadata(
            output_path=f"{os.path.splitext(output_path)[0]}.txt",  # remove ".jpg" from the end and make it ".txt"
            class_id=item_id_tail,
            bbox=item_bbox.to_yolo_bbox(background_copy.width, background_copy.height),
        )


def _overlay_worker(
    background_file: str,
    overlay_area: CocoBbox,
    num_images_to_use: int,
    full_item_dir: str,
    full_background_dir: str,
    full_output_dir: str,
) -> None:
    """
    Worker function to overlay augmented item images onto a single background image.

    This function opens the background image, iterates over all item directories, selects a subset of augmented
    item images, and overlays each of them onto the background image at random positions within the specified
    overlay area. The resulting images and their metadata are saved to the output directory.

    Args:
        background_file (str): The filename of the background image.
        overlay_area (Bbox): The bounding box area within which items can be overlaid.
        num_images_to_use (int): The number of images to use for each item per background (randomly selected).
        full_item_dir (str): The directory containing augmented item images. (ex. data/items)
        full_background_dir (str): The directory containing background images. (ex. data/isaac_backgrounds)
        full_output_dir (str): The directory where processed images will be saved. (ex. data/overlays)
    """
    # full_background_path is like "data/isaac_backgrounds/Library_7.jpg"
    full_background_path = os.path.join(full_background_dir, background_file)
    background_img = Image.open(full_background_path)

    # items are represented by the tail of their item id, these are just numbers like "145".
    item_dirs = os.listdir(full_item_dir)
    for item_dir in item_dirs:
        # concatenated_item_dir is like 'data/items/145/'
        concatenated_item_dir = os.path.join(full_item_dir, item_dir)

        # fmt: off
        # augmented_images is like ["rotate_1234.png", "flip_brightness_5678.png", ...]
        full_augmented_image_paths = [
            os.path.join(concatenated_item_dir, f)
            for f in os.listdir(concatenated_item_dir)
            if f.lower().endswith("png")
        ]  # fmt: on

        # if we generate combos of ALL augmented images and ALL backgrounds the dataset will be like 500gb LOL
        # so let's take a subset. do a random shuffle then select only num_images_to_use of those.
        random.shuffle(full_augmented_image_paths)
        subset_of_image_paths = full_augmented_image_paths[:num_images_to_use]

        _overlay_augmented_images_on_background(
            overlay_area=overlay_area,
            background=background_img,
            item_paths=subset_of_image_paths,
            item_id_tail=item_dir,
            background_name=os.path.splitext(background_file)[0],
            full_output_dir=full_output_dir,
        )


class ImageOverlayProcessor:
    """Repsonsible for overlaying Isaac item images onto various backgrounds."""

    def __init__(self, data_dir: str, background_dir: str, item_dir: str, output_dir: str) -> None:
        """
        Attributes:
            data_dir (str): The root directory for all data.
            background_dir (str): The directory under data_dir containing background images.
            item_dir (str): The directory under data_dir containing augmented item images.
            output_dir (str): The directory under data_dir where processed images will be saved.

        Example:\n`processor = ImageOverlayProcessor(
                data_dir='data',
                background_dir='backgrounds',
                item_dir='items',
                output_dir='overlays'
        )`

        This constructor represents data/backgrounds/ (must exist), data/items/ (must exist),
            and 'data/overlays/' (will be created in this class)
        """
        self.data_dir = data_dir
        self.background_dir = background_dir
        self.item_dir = item_dir
        self.output_dir = output_dir
        self._full_background_dir = os.path.join(self.data_dir, self.background_dir)
        self._full_item_dir = os.path.join(self.data_dir, self.item_dir)
        self._full_output_dir = os.path.join(self.data_dir, self.output_dir)
        os.makedirs(self._full_output_dir, exist_ok=True)

    def _resize_and_pad_image(self, image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        """
        Resize and pad the image to fit the target size while maintaining aspect ratio.

        Args:
            image (PIL.Image.Image): The image to resize and pad.
            target_size (tuple[int, int]): The target size (width, height) for the output image.

        Returns:
            PIL.Image.Image: The resized and padded image.
        """
        # resize the image, preserving aspect ratio
        original_size = image.size
        ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        # pad the image to the target size
        delta_w = target_size[0] - new_size[0]
        delta_h = target_size[1] - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        new_image = ImageOps.expand(image, padding)

        return new_image

    def resize_all_backgrounds(self, target_size: tuple[int, int]) -> None:
        """
        Resize and pad all background images to the target size.

        Args:
            target_size (tuple[int, int]): The target size (width, height) for the output images.
        """
        for filename in os.listdir(self._full_background_dir):
            if filename.lower().endswith(("png", "jpg", "jpeg")):
                image_path = os.path.join(self._full_background_dir, filename)
                image = Image.open(image_path)
                resized_image = self._resize_and_pad_image(image, target_size)
                output_path = os.path.join(self._full_output_dir, filename)
                resized_image.save(output_path)
                logger.debug("Processed and saved: %s to %s", output_path, target_size)

    def visualize_bbox_area(self, background: str | ImageFile.ImageFile, bbox: CocoBbox) -> None:
        """Displays an image with a bounding box overlay.

        Used for debugging purposes. Isaac backgrounds include walls, and we don't
        want to place items on the walls since they don't appear there in game.
        This function helps visualize different bboxes over a background so we can iterate
        and pick an "overlayable" area.

        Args:
            background_name (Union[str, PIL.Image.Image]): The name (not path) of the background file or the image object itself.
            bbox (CocoBbox): Bounding box to overlay on the image.
        """
        image: ImageFile.ImageFile
        if isinstance(background, str):
            image = Image.open(os.path.join(self._full_background_dir, background))
        else:
            image = background
        _, ax = plt.subplots(1)
        ax.imshow(image)
        rect = patches.Rectangle((bbox.x, bbox.y), bbox.w, bbox.h, linewidth=1, edgecolor="r", facecolor=(0, 0, 0, 0))
        ax.add_patch(rect)
        plt.title(os.path.basename(image.filename))
        plt.show()

    def overlay_items_on_backgrounds(
        self, overlay_area: CocoBbox, num_images_to_use: int, seed: int | None = None
    ) -> None:
        """
        Overlay augmented item images (.png) onto background images (.jpg) and save the resulting images as .jpg.

        This method works in parallel across processors using ProcessPoolExecutor.

        Args:
            overlay_area (CocoBbox): The bounding box area within which items can be overlaid.
            num_images_to_use (int): The number of images to use for each item per background (randomly selected).
            seed (int, optional): Seed the randomizer for placing images on backgrounds.
        """
        # Check if the output directory already contains subdirectories like overlays/Library_7/ etc.
        sub_dirs_in_full_output_dir = [
            f for f in os.listdir(self._full_output_dir) if os.path.isdir(os.path.join(self._full_output_dir, f))
        ]

        if sub_dirs_in_full_output_dir:
            confirm = (
                input(
                    f"Looks like there are {len(sub_dirs_in_full_output_dir)} subdirectories in {self._full_output_dir} already. "
                    "Continuing could overwrite existing overlays. Do you want to proceed with overlay generation? (y/n): "
                )
                .strip()
                .lower()
            )

            if confirm not in ["y", "yes"]:
                print("Skipping Overlay process.")
                return

        logger.info("overlay_items_on_backgrounds: Overlaying items on backgrounds, this may take a while...")
        random.seed(seed)

        # backgrounds is a list of filenames such as ["Library_7.jpg", ...]
        backgrounds = [f for f in os.listdir(self._full_background_dir) if f.lower().endswith((".jpeg", ".jpg"))]

        # paralellize: each process will handle all the overlays for one background
        with ProcessPoolExecutor() as executor:
            executor.map(
                _overlay_worker,
                backgrounds,
                repeat(overlay_area),
                repeat(num_images_to_use),
                repeat(self._full_item_dir),
                repeat(self._full_background_dir),
                repeat(self._full_output_dir),
            )

        logger.info("overlay_items_on_backgrounds: Done! Overlaid items onto backgrounds.")

    def plot_random_overlays_with_bboxes(self, num_images: int = 5, seed: int | None = None) -> None:
        """Plot a specified number of overlayed images along with their bounding boxes for verification.

        Basically, go into a random dir in the output dir (i.e. data/overlays/) then plot an image with its bbox.
        Repeat this `num_images` times. Prerequisite: the output dir ) must be populated with images (meaning you have already called
        overlay_items_on_backgrounds(...) before this method.)

        Note: Since the bboxes are stored in .txt files

        So we could plot 1 image from 145/, 1 from 72/, and so on.

        Args:
            num_images (int): The number of images to plot.
            seed (int, optional): Seed for the randomizer.
        """
        random.seed(seed)

        # get all background subdirectories in the data/overlay directory
        backgrounds = os.listdir(self._full_output_dir)

        for _ in range(num_images):
            background = random.choice(backgrounds)
            background_dir = os.path.join(self._full_output_dir, background)
            image_files = [f for f in os.listdir(background_dir) if f.endswith(".jpg")]

            if not image_files:
                continue

            image_file = random.choice(image_files)
            image_path = os.path.join(background_dir, image_file)
            yolo_label_file_path = os.path.splitext(image_path)[0] + ".txt"

            if not os.path.exists(yolo_label_file_path):
                continue

            _, yolo_bbox = read_yolo_label_file(yolo_label_file_path)
            image = Image.open(image_path)
            bbox = yolo_bbox.to_coco_bbox(image.width, image.height)
            self.visualize_bbox_area(image, bbox)


def main():  # pylint: disable=missing-function-docstring
    # example usage, instantiate a processor:
    processor = ImageOverlayProcessor(
        data_dir=DATA_DIR, background_dir=BACKGROUND_DIR, item_dir=ITEM_DIR, output_dir=OVERLAY_DIR
    )

    # you can resize all the backgrounds like this, though they already come resized to TARGET_BACKGROUND_SIZE so won't do it here.
    # processor.resize_all_backgrounds(target_size=TARGET_BACKGROUND_SIZE)

    # generate the overlay/ dataset. Note: this call will take like 6gb up on your drive and will hog cpu for a min or two.
    processor.overlay_items_on_backgrounds(overlay_area=OVERLAYABLE_AREA, num_images_to_use=4, seed=SEED)

    # let's make sure it worked by plotting some images and their bbox
    processor.plot_random_overlays_with_bboxes(num_images=10)


if __name__ == "__main__":
    main()
