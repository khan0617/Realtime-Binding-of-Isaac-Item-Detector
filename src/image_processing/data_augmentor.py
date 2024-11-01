"""
data_augmentor.py provides utilities to generate augmentations 
on the Isaac Item Images to increase our dataset size.
"""

import logging
import os
import random
import uuid
from collections.abc import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from constants import DATA_DIR, ITEM_DIR, SEED, UNMODIFIED_FILE_NAME
from image_processing.augmentation import Augmentation
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def augment_image(
    image_path: str,
    output_dir: str,
    augmentations: Iterable[Iterable[Augmentation] | Augmentation] | None = None,
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
            Ex: "data/items/145"
        augmentations (Iterable[Union[Augmentation, Iterable[Augmentation]]], optional): An iterable of Augmentation or Iterables of
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

    # may need to do a convert() otherwise numpy doesn't preserve the png transparent backround and everything turns black.
    _, file_extension = os.path.splitext(image_path)
    if file_extension in [".png", ".PNG"]:
        image_pil = Image.open(image_path).convert(mode="RGBA")
    else:
        image_pil = Image.open(image_path)
    image_np = np.asarray(image_pil)

    for aug in augmentations:
        if isinstance(aug, Iterable) and not isinstance(aug, str):
            _apply_and_save_combined_augmentations(image_np, output_dir, aug, num_augmented)
        else:
            _apply_and_save_single_augmentation(image_np, output_dir, aug, num_augmented)


def _apply_and_save_combined_augmentations(
    image: np.ndarray, output_dir: str, augmentations: Iterable[Augmentation], num_augmented: int
) -> None:
    """Applies combined augmentations to an image and saves the result.

    This method applies augmentations in sequence to the provided image. If all augmentations
    in the iterable are operations that do not produce unique results when repeated, it applies the
    augmentations only once. Otherwise, it applies the sequence multiple times as specified by `num_augmented`.

    Args:
        image (np.ndarray): The original image to be augmented.
        output_dir (str): The directory where the augmented images will be saved.
        augmentations (Iterable[Augmentation, ...]): A collection of Augmentation enum values indicating the sequence of augmentations to apply.
        num_augmented (int): The number of times to apply the sequence of augmentations. This value is overridden to 1 if all augmentations
                            are in the non-repeating category.
    """
    if all(sub_aug in Augmentation.operations_to_not_repeat() for sub_aug in augmentations):
        _apply_and_save_helper(image, output_dir, augmentations, num_augmented=1)
    else:
        _apply_and_save_helper(image, output_dir, augmentations, num_augmented)


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
        _apply_and_save_helper(image, output_dir, (augmentation,), num_augmented=1)
    else:
        _apply_and_save_helper(image, output_dir, (augmentation,), num_augmented)


def _apply_and_save_helper(
    image: np.ndarray, output_dir: str, augmentations: Iterable[Augmentation], num_augmented: int
) -> None:
    """Applies specified augmentations to an image and saves the augmented images.

    Args:
        image (np.ndarray): The original image to be augmented.
        output_dir (str): The directory where the augmented images will be saved.
        augmentations (Iterable[Augmentation, ...]): A collection of Augmentation enum values specifying the augmentations to apply.
        num_augmented (int): The number of augmented versions of the image to generate.
    """
    for _ in range(num_augmented):
        augmented_image = image.copy()
        for aug in augmentations:
            augmented_image = _dispatch_augmentation(augmented_image, aug)
        _save_image(augmented_image, output_dir, augmentations)


def _save_image(image: np.ndarray, output_dir: str, augmentations: Iterable[Augmentation]) -> None:
    """Saves the augmented image to the specified directory with a descriptive filename.

    Images are saved with the list of augmentations applied to them in order. For example, if an image is
    1st rotated, then had color jitter, then noise last, the image will be saved as:
    rotate_color_jitter_noise_1234.png, where 1234 are the 1st 4 digits of a random uuid. The general template is:
    `{augmentation1_augmentation2_..._augmentationX}_{4 digits of uuid}.png`

    Args:
        image (np.ndarray): The augmented image as a NumPy array.
        output_dir (str): The directory where the image will be saved.
        augmentations (Iterable[Augmentation, ...]): A collection of Augmentation enum values specifying the augmentations applied.
    """
    image_pil = Image.fromarray(image)
    augmentations_applied = "_".join(aug.lower() for aug in augmentations)
    filename = (
        f"{augmentations_applied}_{uuid.uuid4().hex[:4]}.png"  # apply a random uuid prefix to each image for uniqueness
    )
    filepath = os.path.join(output_dir, filename)
    logger.debug("_save_image: Saving img to %s, image size is: %s", filepath, image_pil.size)
    image_pil.save(filepath)


# pylint: disable=too-many-return-statements, too-many-branches
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
        return _rotate(image)

    elif augmentation is Augmentation.NOISE:
        return _noise(image)

    elif augmentation is Augmentation.VERTICAL_FLIP:
        return _vertical_flip(image)

    elif augmentation is Augmentation.HORIZONTAL_MIRROR:
        return _horizontal_mirror(image)

    elif augmentation is Augmentation.BRIGHTNESS:
        return _brightness(image)

    elif augmentation is Augmentation.CONTRAST:
        return _contrast(image)

    elif augmentation is Augmentation.SCALE:
        return _scale(image)

    elif augmentation is Augmentation.TRANSLATE:
        return _translate(image)

    elif augmentation is Augmentation.SHEAR:
        return _shear(image)

    elif augmentation is Augmentation.COLOR_JITTER:
        return _color_jitter(image)

    elif augmentation is Augmentation.SHARPNESS:
        return _sharpness(image)

    elif augmentation is Augmentation.SMOOTH:
        return _smooth(image)

    else:
        raise ValueError(f"Unimplemented Augmentation: {augmentation}")


def _rotate(image: np.ndarray) -> np.ndarray:
    """Rotates the image by a random angle in the range +-170 degrees"""
    angle = np.random.uniform(-170, 170)
    image_pil = Image.fromarray(image)
    rotated_image = image_pil.rotate(angle, resample=Image.Resampling.BICUBIC)
    return np.asarray(rotated_image)


def _noise(image: np.ndarray) -> np.ndarray:
    """Adds random Gaussian noise to the image."""
    noise_level = np.random.uniform(0.01, 0.05)
    if image.shape[-1] == 4:  # RGBA image, let's not corrupt the alpha (transparency) channel.
        rgb, alpha = image[..., :3], image[..., 3:]
        noise = np.random.normal(0, 255 * noise_level, rgb.shape)  # apply noise only to RGB
        noisy_rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)
        return np.concatenate((noisy_rgb, alpha), axis=-1)
    else:  # RGB image
        noise = np.random.normal(0, 255 * noise_level, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image  # type: ignore


def _vertical_flip(image: np.ndarray) -> np.ndarray:
    """Flip the image vertically (top to bottom flip)"""
    flipped_pil_img = ImageOps.flip(Image.fromarray(image))
    return np.asarray(flipped_pil_img)


def _horizontal_mirror(image: np.ndarray) -> np.ndarray:
    "Mirror the image horizontally (left to right flip)"
    mirrored_pil_img = ImageOps.mirror(Image.fromarray(image))
    return np.asarray(mirrored_pil_img)


def _brightness(image: np.ndarray) -> np.ndarray:
    """Adjusts the brightness of the image with a random factor in the range [0.5, 1.5]."""
    factor = np.random.uniform(0.5, 1.5)
    image_pil = Image.fromarray(image)
    bright_image = ImageEnhance.Brightness(image_pil).enhance(factor)
    return np.asarray(bright_image)


def _contrast(image: np.ndarray) -> np.ndarray:
    """Adjusts the contrast of the image with a random factor in the range [0.5, 1.5]."""
    factor = np.random.uniform(0.5, 1.5)
    image_pil = Image.fromarray(image)
    contrast_image = ImageEnhance.Contrast(image_pil).enhance(factor)
    return np.asarray(contrast_image)


def _scale(image: np.ndarray) -> np.ndarray:
    """Scales the image randomly while keeping the original size, using a scaling factor in range [0.8, 1.2]"""
    scale_factor = np.random.uniform(0.8, 1.2)
    image_pil = Image.fromarray(image)
    scaled_image = ImageOps.scale(image_pil, scale_factor)
    scaled_image_back_to_original_size = ImageOps.fit(scaled_image, image_pil.size)
    return np.asarray(scaled_image_back_to_original_size)


def _translate(image: np.ndarray) -> np.ndarray:
    """Translates the image randomly along the x and y axis up to 20% of img dimensions."""
    max_trans = 0.2
    width, height = image.shape[1], image.shape[0]

    # build the translation matrix
    dx = int(np.random.uniform(-max_trans, max_trans) * width)
    dy = int(np.random.uniform(-max_trans, max_trans) * height)

    # fmt:off
    # the first 3 elements represent the changing of x, last 3 represent changing y
    translation_matrix = (1, 0, dx, 0, 1, dy)  # https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations 
    # fmt: on

    image_pil = Image.fromarray(image)
    translated_image = image_pil.transform(image_pil.size, Image.Transform.AFFINE, translation_matrix)
    return np.asarray(translated_image)


def _shear(image: np.ndarray) -> np.ndarray:
    """Shears the image randomly with a shear factor in range [-0.3, 0.3]."""
    shear_factor = np.random.uniform(-0.3, 0.3)
    image_pil = Image.fromarray(image)
    shear_matrix = (1, shear_factor, 0, 0, 1, 0)  # https://mathworld.wolfram.com/ShearMatrix.html
    sheared_image = image_pil.transform(image_pil.size, Image.Transform.AFFINE, shear_matrix)
    return np.asarray(sheared_image)


def _color_jitter(image: np.ndarray) -> np.ndarray:
    """Randomly adjust color balance of the image, with a factor in range [0.3, 1.0]."""
    color_factor = np.random.uniform(0.3, 1.0)
    image_pil = Image.fromarray(image)
    jittered_image = ImageEnhance.Color(image_pil).enhance(color_factor)
    return np.asarray(jittered_image)


def _sharpness(image: np.ndarray) -> np.ndarray:
    """Adjusts the sharpness of the image with a random factor in the range [0.5, 1.5]"""
    factor = np.random.uniform(0.5, 1.5)
    image_pil = Image.fromarray(image)
    sharpened_image = ImageEnhance.Sharpness(image_pil).enhance(factor)
    return np.asarray(sharpened_image)


def _smooth(image: np.ndarray) -> np.ndarray:
    """Smooth the image, applying the following smoothing filter kernel from PIL.

    filterargs = (3, 3), 13, 0, (
        1, 1, 1,
        1, 5, 1,
        1, 1, 1,
    )
    """
    image_pil = Image.fromarray(image)
    filtered_image = image_pil.filter(ImageFilter.SMOOTH)
    return np.asarray(filtered_image)


def main() -> None:
    # example usage. We'll apply every augmentation to the "8_Inch_Nails" item.
    item_id = "359"
    image_path = os.path.join(DATA_DIR, ITEM_DIR, item_id, UNMODIFIED_FILE_NAME)
    output_dir = os.path.join("new_dir", item_id)

    # produce 1 image for each augmentation.
    augment_image(image_path, output_dir, num_augmented=1, seed=SEED)


if __name__ == "__main__":
    main()
