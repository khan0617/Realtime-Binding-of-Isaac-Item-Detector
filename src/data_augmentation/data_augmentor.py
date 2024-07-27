import os
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from data_augmentation.augmentation import Augmentation


class DataAugmentor:
    """DataAugmentor provides augmentations on the Isaac Item Images to increase our dataset size."""

    @staticmethod
    def augment_image(
        image_path: str,
        output_dir: str,
        augmentations: list[str] | None = None,
        num_augmented: int = 15,
        seed: int | None = None,
    ) -> None:
        """Augments an image with various transformations and saves the augmented images to the specified directory.

        This function applies a series of augmentation techniques to the input image, including rotation, flipping,
        and adding noise. The augmented images are saved in the output directory with descriptive filenames indicating
        the type of augmentation applied.

        Args:
            image_path (str): The file path of the original image to be augmented.
            output_dir (str): The directory where the augmented images will be saved. The directory must exist.
            augmentations (list[Auugmentation], optional): A list of Augmentation specifying which augmentations to apply.
                If None, a default set of all augmentations is applied.
            num_augmented (int, optional): The number of augmented images to generate per augmentation technique.
                Ex: If num_augmented is 15, and you pass in augmentations=[Augmentation.ROTATE] we generate 15 rotated images at random degrees.
                num_augmented only applies to augmentations which make sense to have multiple (i.e. not FLIP).
            seed (int, optional): Random seed for reproducibility. If None, the seed is not set.
        """
        if augmentations is None:
            augmentations = list(Augmentation)  # use all Enum values by default.

        if seed is not None:
            random.seed(seed)

        image = Image.open(image_path)
        image_np = np.array(image)

        for aug in augmentations:
            if aug is Augmentation.ROTATE:
                for i in range(num_augmented):
                    pass

            elif aug is Augmentation.NOISE:
                for i in range(num_augmented):
                    pass

            elif aug is Augmentation.FLIP:
                pass

            elif aug is Augmentation.MIRROR:
                pass

            elif aug is Augmentation.BRIGHTNESS:
                pass

            elif aug is Augmentation.CONTRAST:
                pass

            elif aug is Augmentation.SCALE:
                pass

            elif aug is Augmentation.TRANSLATE:
                pass

            elif aug is Augmentation.SHEAR:
                pass

            elif aug is Augmentation.COLOR_JITTER:
                pass

            else:
                raise ValueError(f"Unimplemented Augmentation: {aug}")
