from unittest.mock import Mock, patch

from pygetwindow import Win32Window

from src.image_processing.screen_grabber import ScreenGrabber


@patch("sys.exit")
@patch("src.image_processing.screen_grabber.ScreenGrabber._get_isaac_window", return_value=None)
def test_screen_grabber_constructor_without_isaac_running(mock_get_window: Mock, mock_sys_exit: Mock) -> None:
    screen_grabber = ScreenGrabber()
    mock_sys_exit.assert_called_once_with(1)


@patch("src.image_processing.screen_grabber.gw.getAllTitles", return_value=["Chrome", "Binding of Isaac: Repentance"])
@patch("src.image_processing.screen_grabber.gw.getWindowsWithTitle", return_value=[Mock(spec=Win32Window)])
def test_get_isaac_window(mock_get_windows_with_title: Mock, mock_get_all_titles: Mock) -> None:
    # given/when
    screen_grabber = ScreenGrabber()

    # then
    mock_get_all_titles.assert_called_once()
    mock_get_windows_with_title.assert_called_once_with("Binding of Isaac: Repentance".lower())
    assert isinstance(screen_grabber._isaac_window, Win32Window)
