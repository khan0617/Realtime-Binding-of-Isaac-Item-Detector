from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.constants import UNMODIFIED_FILE_NAME
from src.image_processing.data_augmentor import augment_image


@pytest.fixture
def file_path() -> Path:
    file_path = Path(__file__).parent.absolute().parent.absolute() / "test_resources" / UNMODIFIED_FILE_NAME
    return file_path


@patch("src.image_processing.data_augmentor._rotate", return_value=np.zeros((64, 64, 4), dtype=np.uint8))
@patch("src.image_processing.data_augmentor._noise", return_value=np.zeros((64, 64, 4), dtype=np.uint8))
@patch("src.image_processing.data_augmentor._vertical_flip", return_value=np.zeros((64, 64, 4), dtype=np.uint8))
@patch("src.image_processing.data_augmentor._horizontal_mirror", return_value=np.zeros((64, 64, 4), dtype=np.uint8))
@patch("src.image_processing.data_augmentor._brightness", return_value=np.zeros((64, 64, 4), dtype=np.uint8))
@patch("src.image_processing.data_augmentor._contrast", return_value=np.zeros((64, 64, 4), dtype=np.uint8))
@patch("src.image_processing.data_augmentor._scale", return_value=np.zeros((64, 64, 4), dtype=np.uint8))
@patch("src.image_processing.data_augmentor._translate", return_value=np.zeros((64, 64, 4), dtype=np.uint8))
@patch("src.image_processing.data_augmentor._shear", return_value=np.zeros((64, 64, 4), dtype=np.uint8))
@patch("src.image_processing.data_augmentor._color_jitter", return_value=np.zeros((64, 64, 4), dtype=np.uint8))
@patch("src.image_processing.data_augmentor._sharpness", return_value=np.zeros((64, 64, 4), dtype=np.uint8))
@patch("src.image_processing.data_augmentor._smooth", return_value=np.zeros((64, 64, 4), dtype=np.uint8))
def test_augment_image_calls_all_augmentations(
    mock_smooth: Mock,
    mock_sharpness: Mock,
    mock_color_jitter: Mock,
    mock_shear: Mock,
    mock_translate: Mock,
    mock_scale: Mock,
    mock_contrast: Mock,
    mock_brightness: Mock,
    mock_horizontal_mirror: Mock,
    mock_vertical_flip: Mock,
    mock_noise: Mock,
    mock_rotate: Mock,
    file_path: Path,
    tmp_path: Path,
):
    """Test that each augmentation function is called once when passed to augment_image."""
    # given/when
    augment_image(image_path=file_path, output_dir=tmp_path, num_augmented=1)

    # then
    mock_rotate.assert_called_once()
    mock_noise.assert_called_once()
    mock_vertical_flip.assert_called_once()
    mock_horizontal_mirror.assert_called_once()
    mock_brightness.assert_called_once()
    mock_contrast.assert_called_once()
    mock_scale.assert_called_once()
    mock_translate.assert_called_once()
    mock_shear.assert_called_once()
    mock_color_jitter.assert_called_once()
    mock_sharpness.assert_called_once()
    mock_smooth.assert_called_once()

    # Check that the augmented images were saved to the temp directory
    augmented_images = list(tmp_path.glob("*.png"))
    assert len(augmented_images) > 0, "No augmented images were saved to the directory."
