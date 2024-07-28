import os
import random
import uuid

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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
        np.random.seed(seed)

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
        """Applies combined augmentations to an image and saves the result.

        This method applies a tuple of augmentations in sequence to the provided image. If all augmentations
        in the tuple are operations that do not produce unique results when repeated, it applies the
        augmentations only once. Otherwise, it applies the sequence multiple times as specified by `num_augmented`.

        Args:
            image (np.ndarray): The original image to be augmented.
            output_dir (str): The directory where the augmented images will be saved.
            augmentations (tuple[Augmentation, ...]): A tuple of Augmentation enum values indicating the sequence of augmentations to apply.
            num_augmented (int): The number of times to apply the sequence of augmentations. This value is overridden to 1 if all augmentations
                                are in the non-repeating category.
        """
        if all(sub_aug in Augmentation.operations_to_not_repeat() for sub_aug in augmentations):
            DataAugmentor._apply_and_save_helper(image, output_dir, augmentations, num_augmented=1)
        else:
            DataAugmentor._apply_and_save_helper(image, output_dir, augmentations, num_augmented)

    @staticmethod
    def _apply_and_save_single_augmentation(
        image: np.ndarray, output_dir: str, augmentation: Augmentation, num_augmented: int
    ) -> None:
        """Applies a single augmentation to an image and saves the result.

        This method applies a specified augmentation to the provided image. If the augmentation is categorized
        as non-repeating (i.e., it does not produce unique results when applied multiple times), it applies the
        augmentation only once. Otherwise, it applies the augmentation multiple times as specified by `num_augmented`.

        Args:
            image (np.ndarray): The original image to be augmented.
            output_dir (str): The directory where the augmented images will be saved.
            augmentation (Augmentation): An Augmentation enum value specifying the augmentation to apply.
            num_augmented (int): The number of times to apply the augmentation. This value is overridden to 1 if the augmentation
                                is in the non-repeating category.
        """
        if augmentation in Augmentation.operations_to_not_repeat():
            DataAugmentor._apply_and_save_helper(image, output_dir, (augmentation,), num_augmented=1)
        else:
            DataAugmentor._apply_and_save_helper(image, output_dir, (augmentation,), num_augmented)

    @staticmethod
    def _apply_and_save_helper(
        image: np.ndarray, output_dir: str, augmentations: tuple[Augmentation, ...], num_augmented: int
    ) -> None:
        """Applies specified augmentations to an image and saves the augmented images.

        Args:
            image (np.ndarray): The original image to be augmented.
            output_dir (str): The directory where the augmented images will be saved.
            augmentations (tuple[Augmentation, ...]): A tuple of Augmentation enum values specifying the augmentations to apply.
            num_augmented (int): The number of augmented versions of the image to generate.
        """
        for _ in range(num_augmented):
            augmented_image = image.copy()
            for aug in augmentations:
                augmented_image = DataAugmentor._dispatch_augmentation(augmented_image, aug)
            DataAugmentor._save_image(augmented_image, output_dir, augmentations)

    @staticmethod
    def _save_image(image: np.ndarray, output_dir: str, augmentations: tuple[Augmentation, ...]) -> None:
        """Saves the augmented image to the specified directory with a descriptive filename.

        Images are saved with the list of augmentations applied to them in order. For example, if an image is
        1st rotated, then had color jitter, then noise last, the image will be saved as:
        rotate_color_jitter_noise_1234.png, where 1234 are the 1st 4 digits of a random uuid. The general template is:
        `{augmentation1_augmentation2_..._augmentationX}_{4 digits of uuid}.png`

        Args:
            image (np.ndarray): The augmented image as a NumPy array.
            output_dir (str): The directory where the image will be saved.
            augmentations (Tuple[Augmentation, ...]): A tuple of Augmentation enum values specifying the augmentations applied.
        """
        image_pil = Image.fromarray(image)
        augmentations_applied = "_".join(aug.lower() for aug in augmentations)
        filename = f"{augmentations_applied}_{uuid.uuid4().hex[:4]}.png"  # apply a random
        filepath = os.path.join(output_dir, filename)
        image_pil.save(filepath)

    @staticmethod
    def _dispatch_augmentation(image: np.ndarray, augmentation: Augmentation) -> np.ndarray:
        """Dispatches the given augmentation to the appropriate method.

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

        elif augmentation is Augmentation.SHARPNESS:
            return DataAugmentor._sharpness(image)

        else:
            raise ValueError(f"Unimplemented Augmentation: {augmentation}")

    @staticmethod
    def _rotate(image: np.ndarray) -> np.ndarray:
        """Rotates the image by a random angle +-60 degrees"""
        angle = np.random.uniform(-60, 60)
        image_pil = Image.fromarray(image)
        rotated_image = image_pil.rotate(angle, resample=Image.Resampling.BICUBIC)
        return np.array(rotated_image)

    @staticmethod
    def _noise(image: np.ndarray) -> np.ndarray:
        """Adds random Gaussian noise to the image."""
        noise_level = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, 255 * noise_level, image.shape)  # gaussian centered at 0
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    @staticmethod
    def _flip(image: np.ndarray) -> np.ndarray:
        """Flip the imnage vertically (top to bottom flip)"""
        flipped_pil_img = ImageOps.flip(Image.fromarray(image))
        return np.array(flipped_pil_img)

    @staticmethod
    def _mirror(image: np.ndarray) -> np.ndarray:
        "Mirror the image horizontally (left to right flip)"
        mirrored_pil_img = ImageOps.mirror(Image.fromarray(image))
        return np.array(mirrored_pil_img)

    @staticmethod
    def _brightness(image: np.ndarray) -> np.ndarray:
        """Adjusts the brightness of the image with a random factor in the range [0.5, 1.5]."""
        factor = np.random.uniform(0.5, 1.5)
        image_pil = Image.fromarray(image)
        bright_image = ImageEnhance.Brightness(image_pil).enhance(factor)
        return np.array(bright_image)

    @staticmethod
    def _contrast(image: np.ndarray) -> np.ndarray:
        """Adjusts the contrast of the image with a random factor in the range [0.5, 1.5]."""
        factor = np.random.uniform(0.5, 1.5)
        image_pil = Image.fromarray(image)
        contrast_image = ImageEnhance.Contrast(image_pil).enhance(factor)
        return np.array(contrast_image)

    @staticmethod
    def _scale(image: np.ndarray) -> np.ndarray:
        """Scales the image randomly while keeping the original size, using a scaling factor in range [0.8, 1.2]"""
        scale_factor = np.random.uniform(0.8, 1.2)
        image_pil = Image.fromarray(image)
        scaled_image = ImageOps.scale(image_pil, scale_factor)
        scaled_image_back_to_original_size = ImageOps.fit(scaled_image, image_pil.size)
        return np.array(scaled_image_back_to_original_size)

    @staticmethod
    def _translate(image: np.ndarray) -> np.ndarray:
        """Translates the image randomly along the x and y axis up to 20% of img dimensions."""
        max_trans = 0.2
        width, height = image.shape[1], image.shape[0]

        # build the translation matrix
        dx = int(np.random.uniform(-max_trans, max_trans) * width)
        dy = int(np.random.uniform(-max_trans, max_trans) * height)

        # the first 3 elements represent the changing of x, last 3 represent changing y.
        translation_matrix = (
            1,
            0,
            dx,
            0,
            1,
            dy,
        )  # https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
        image_pil = Image.fromarray(image)
        translated_image = image_pil.transform(image_pil.size, Image.Transform.AFFINE, translation_matrix)
        return np.array(translated_image)

    @staticmethod
    def _shear(image: np.ndarray) -> np.ndarray:
        """Shears the image randomly with a shear factor in range [-0.3, 0.3]."""
        shear_factor = np.random.uniform(-0.3, 0.3)
        image_pil = Image.fromarray(image)
        shear_matrix = (1, shear_factor, 0, 0, 1, 0)  # https://mathworld.wolfram.com/ShearMatrix.html
        sheared_image = image_pil.transform(image_pil.size, Image.Transform.AFFINE, shear_matrix)
        return np.array(sheared_image)

    @staticmethod
    def _color_jitter(image: np.ndarray) -> np.ndarray:
        """Randomly adjust color balance of the image, with a factor in range [0.5, 1.0]."""
        color_factor = np.random.uniform(0.5, 1.0)
        image_pil = Image.fromarray(image)
        jittered_image = ImageEnhance.Color(image_pil).enhance(color_factor)
        return np.array(color_factor)

    @staticmethod
    def _sharpness(image: np.ndarray) -> np.ndarray:
        """Adjusts the sharpness of the image with a random factor in the range [0.5, 1.5]"""
        factor = np.random.uniform(0.5, 1.5)
        image_pil = Image.fromarray(image)
        sharpened_image = ImageEnhance.Sharpness(image_pil).enhance(factor)
        return np.array(sharpened_image)
