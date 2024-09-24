from unittest.mock import Mock, patch

from src.logging_config import configure_logging


@patch("src.logging_config.logging.basicConfig")
@patch("src.logging_config.logging.FileHandler")
@patch("src.logging_config.logging.StreamHandler")
def test_configure_logging_called_once(
    mock_stream_handler: Mock, mock_file_handler: Mock, mock_basic_config: Mock
) -> None:
    # given/when
    configure_logging()

    # then
    mock_stream_handler.assert_called_once()
    mock_basic_config.assert_called_once()
    mock_file_handler.assert_called_once()

    # when (round 2)
    configure_logging()

    # then (round 2)
    mock_basic_config.assert_called_once()  # basicConfig should only be called once.
