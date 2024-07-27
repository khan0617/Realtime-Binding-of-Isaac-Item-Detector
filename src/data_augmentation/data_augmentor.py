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
        augmentations: list[tuple[Augmentation, ...] | Augmentation] | None = None,
        num_augmented: int = 15,
        seed: int | None = None,
    ) -> None:
        """Augments an image with various transformations and saves the augmented images to the specified directory.

        This function applies a series of augmentation techniques to the input image, including rotation, flipping,
        and adding noise among others. The augmented images are saved in the output directory with descriptive filenames indicating
        the type(s) and order of augmentation(s) applied.

        Args:
            image_path (str): The file path of the original image to be augmented.
            output_dir (str): The directory where the augmented images will be saved. The directory will be created if it doesn't exist.
            augmentations (list[Union[Augmentation, Tuple[Augmentation]]], optional): A list of Augmentation or tuples of
                Augmentations specifying which augmentations to apply. If None, a default set of all augmentations is applied.
                Ex: If you pass in `[(Augmentation.ROTATE, Augmentation.FLIP), Augmentation.NOISE]`, for the first tuple,
                    all augmented images will first be rotated, THEN flipped. For the 2nd item, those images will only have NOISE applied.
                    If num_augmented is 5, then 5 images with ROTATE -> FLIP, and 5 images with just FLIP will be created, for a total of 10.
            num_augmented (int, optional): The number of augmented images to generate per augmentation technique.
                Ex: If num_augmented is 15, and you pass in augmentations=[Augmentation.ROTATE] we generate 15 rotated images at random degrees.
                num_augmented only applies to augmentations which make sense to have multiple (i.e. not FLIP).
            seed (int, optional): Random seed for reproducibility. If None, the seed is not set.
        """
        random.seed(seed)

        if augmentations is None:
            augmentations = list(Augmentation)  # apply all the augmentations by default

        os.makedirs(output_dir, exist_ok=True)
        image_np = np.array(Image.open(image_path))

        for aug in augmentations:
            if isinstance(aug, tuple):
                DataAugmentor._apply_and_save_combined_augmentations(image_np, output_dir, aug, num_augmented)
            else:
                DataAugmentor._apply_and_save_single_augmentation(image_np, output_dir, aug, num_augmented)

    @staticmethod
    def _apply_and_save_combined_augmentations(
        image: np.ndarray, output_dir: str, augmentations: tuple[Augmentation, ...], num_augmented: int
    ) -> None:
        """Applies combined augmentations and saves the result."""
        # TODO docstring
        if all(sub_aug in Augmentation.operations_to_not_repeat() for sub_aug in augmentations):
            # don't repeat the augmentations. For example, it doesn't make sense to do FLIP->MIRROR 5 times, it'll be the same image.
            DataAugmentor._apply_and_save(image, output_dir, augmentations, num_augmented=1)
        else:
            DataAugmentor._apply_and_save(image, output_dir, augmentations, num_augmented)

    @staticmethod
    def _apply_and_save_single_augmentation(
        image: np.ndarray, output_dir: str, augmentation: Augmentation, num_augmented: int
    ) -> None:
        """Applies a single augmentation and saves the result."""
        # TODO docstring
        if augmentation in Augmentation.operations_to_not_repeat():
            DataAugmentor._apply_and_save(image, output_dir, (augmentation,), num_augmented=1)
        else:
            DataAugmentor._apply_and_save(image, output_dir, (augmentation,), num_augmented)

    @staticmethod
    def _apply_and_save(
        image: np.ndarray, output_dir: str, augmentations: tuple[Augmentation, ...], num_augmented: int
    ) -> None:
        """Applies the specified augmentations and saves the images."""
        for _ in range(num_augmented):
            augmented_image = image.copy()
            for aug in augmentations:
                augmented_image = DataAugmentor._augment_image(augmented_image, aug)
            DataAugmentor._save_image()

    @staticmethod
    def _save_image() -> None:
        # TODO
        pass

    @staticmethod
    def _augment_image(image: np.ndarray, augmentation: Augmentation) -> np.ndarray:
        """Applies a single augmentation to an image and returns the augmented image.

        Serves as a dispatcher table for the specified augmentation.

        Args:
            image (np.ndarray): The image to augment as an np array.
            augmentation (Augmentation): The augmentation to apply to the image.

        Returns:
            The augmented image as an np array.
        """
        if augmentation is Augmentation.ROTATE:
            return DataAugmentor._rotate(image)

        elif augmentation is Augmentation.NOISE:
            return DataAugmentor._noise(image)

        elif augmentation is Augmentation.FLIP:
            return DataAugmentor._flip(image)

        elif augmentation is Augmentation.MIRROR:
            return DataAugmentor._mirror(image)

        elif augmentation is Augmentation.BRIGHTNESS:
            return DataAugmentor._brightness(image)

        elif augmentation is Augmentation.CONTRAST:
            return DataAugmentor._contrast(image)

        elif augmentation is Augmentation.SCALE:
            return DataAugmentor._scale(image)

        elif augmentation is Augmentation.TRANSLATE:
            return DataAugmentor._translate(image)

        elif augmentation is Augmentation.SHEAR:
            return DataAugmentor._shear(image)

        elif augmentation is Augmentation.COLOR_JITTER:
            return DataAugmentor._color_jitter(image)

        else:
            raise ValueError(f"Unimplemented Augmentation: {augmentation}")

    @staticmethod
    def _rotate(image: np.ndarray) -> np.ndarray:
        return np.ndarray((1, 1))

    @staticmethod
    def _noise(image: np.ndarray) -> np.ndarray:
        return np.ndarray((1, 1))

    @staticmethod
    def _flip(image: np.ndarray) -> np.ndarray:
        return np.ndarray((1, 1))

    @staticmethod
    def _mirror(image: np.ndarrayr) -> np.ndarray:
        return np.ndarray((1, 1))

    @staticmethod
    def _brightness(image: np.ndarray) -> np.ndarray:
        return np.ndarray((1, 1))

    @staticmethod
    def _contrast(image: np.ndarray) -> np.ndarray:
        return np.ndarray((1, 1))

    @staticmethod
    def _scale(image: np.ndarray) -> np.ndarray:
        return np.ndarray((1, 1))

    @staticmethod
    def _translate(image: np.ndarray) -> np.ndarray:
        return np.ndarray((1, 1))

    @staticmethod
    def _shear(image: np.ndarray) -> np.ndarray:
        return np.ndarray((1, 1))

    @staticmethod
    def _color_jitter(image: np.ndarray) -> np.ndarray:
        return np.ndarray((1, 1))
