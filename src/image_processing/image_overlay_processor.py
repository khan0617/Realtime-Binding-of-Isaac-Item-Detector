import logging
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from constants import BACKGROUND_DIR, DATA_DIR, ITEM_DIR, OVERLAY_DIR, TARGET_BACKGROUND_SIZE
from image_processing.bbox import Bbox
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class ImageOverlayProcessor:
    """Repsonsible for overlaying Isaac item images onto various backgrounds."""

    def __init__(self, data_dir: str, background_dir: str, item_dir: str, output_dir: str) -> None:
        """
        Attributes:
            data_dir (str): The root directory for all data.
            background_dir (str): The directory under data_dir containing background images.
            item_dir (str): The directory under data_dir containing augmented item images.
            output_dir (str): The directory under data_dir where processed images will be saved.

        Example:
            processor = ImageOverlayProcessor(
                data_dir='data',
                background_dir='backgrounds',
                item_dir='items',
                output_dir='overlays'
            )

            # This sets up the processor to read backgrounds from 'data/backgrounds',
            # item images from 'data/items', and save the output to 'data/overlays'.
        """
        self.data_dir = data_dir
        self.background_dir = background_dir
        self.item_dir = item_dir
        self.output_dir = output_dir
        self._full_background_dir = os.path.join(self.data_dir, self.background_dir)
        self._full_item_dir = os.path.join(self.data_dir, self.item_dir)
        self._full_output_dir = os.path.join(self.data_dir, self.output_dir)
        os.makedirs(self._full_output_dir, exist_ok=True)

    def resize_and_pad_image(self, image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
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
                resized_image = self.resize_and_pad_image(image, target_size)
                output_path = os.path.join(self._full_output_dir, filename)
                resized_image.save(output_path)
                logger.debug("Processed and saved: %s to %s", output_path, target_size)

    def visualize_overlayable_area(self, background_name: str, bbox: Bbox) -> None:
        """Displays an image with a bounding box overlay.

        Used for debugging purposes. Isaac backgrounds include walls, and we don't
        want to place items on the walls since they don't appear there in game.
        This function helps visualize different bboxes over a background so we can iterate
        and pick a "overlayable" area.

        Args:
            background_name (str): The name (not path) of the background file.
            bbox (Bbox): Bounding box to overlay on the image.
        """
        image = Image.open(os.path.join(self._full_background_dir, background_name))
        _, ax = plt.subplots(1)
        ax.imshow(image)
        rect = patches.Rectangle((bbox.x, bbox.y), bbox.w, bbox.h, linewidth=1, edgecolor="r", facecolor=(0, 0, 0, 0))
        ax.add_patch(rect)
        plt.show()


def main():  # pylint: disable=missing-function-docstring
    processor = ImageOverlayProcessor(
        data_dir=DATA_DIR, background_dir=BACKGROUND_DIR, item_dir=ITEM_DIR, output_dir=BACKGROUND_DIR
    )

    # reminder imgs are (1000, 625)
    bbox = Bbox(115, 105, 770, 415)
    processor.visualize_overlayable_area("Stage_Cellar_room.jpg", bbox)


if __name__ == "__main__":
    main()
