"""
`screen_grabber.py`

Utilities to find and capture the Isaac window on screen.
Works for Windows operating systems only.
"""

import logging
import sys
from typing import cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygetwindow as gw  # type: ignore
from mss.windows import MSS as mss
from pygetwindow import Win32Window

from constants import ISAAC_WINDOW_TITLE
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class ScreenGrabber:
    """
    Utilities to facilitate screen capturing.
    """

    def __init__(self) -> None:
        self._isaac_window = cast(Win32Window, self._get_isaac_window())
        if not self._isaac_window:
            logger.error("No Isaac window found, exiting.")
            sys.exit(1)

    def _get_isaac_window(self) -> Win32Window | None:
        """
        Get the Window object for the Binding of Isaac.

        Returns:
            The first Win32Window which has a title matching "Binding of Isaac".
            Returns None if no window is found.
        """
        # get all window titles that are not empty strings
        titles = [t.lower() for t in gw.getAllTitles() if t]

        for title in titles:
            if "chrome" not in title and "firefox" not in title:
                if ISAAC_WINDOW_TITLE.lower() in title:
                    windows: list[Win32Window] = gw.getWindowsWithTitle(title)
                    return windows[0]

        logger.warning("_get_isaac_window: Failed to find Binding of Isaac Window! Are you running the game?")
        return None

    def capture_window(self) -> np.ndarray:
        """
        Capture the specific window.
        To capture properly, the window must be in the foreground.

        Returns:
            The captured screen area as an np array.
        """
        with mss() as sct:
            monitor = {
                "top": self._isaac_window.top,
                "left": self._isaac_window.left,
                "width": self._isaac_window.width,
                "height": self._isaac_window.height,
            }
            frame = np.array(sct.grab(monitor))

            # mss captures with the title bar and extra space on the sides in windowed mode
            # let's crop those out.
            title_bar_height = 50
            border_width = 8
            cropped_frame = frame[
                title_bar_height:-border_width,  # crop the top and bottom
                border_width:-border_width,  # crop left and right border
            ]

            # mss captures the image in BGR format, we need to convert it before we can plot it later.
            color_corrected_image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGRA2RGB)  # pylint: disable=no-member
            return color_corrected_image

    def display_captured_window(self, frame: np.ndarray | None = None) -> None:
        """
        Use matplotlib to show the captured frame.

        Args:
            frame (np.ndarray | None, optional): The frame as an np array.
                If None, the frame will be captured in this method.
        """
        try:
            frame = frame if frame is not None else self.capture_window()
            plt.imshow(frame)
            plt.axis("off")
            plt.title("Captured Isaac Window")
            plt.show()
        except RuntimeError as e:
            logger.error("Failed to display captured window: %s", str(e))


def main():
    # before running, make sure your Isaac game window is running + in the foreground
    screen_grabber = ScreenGrabber()
    screen_grabber.display_captured_window()


if __name__ == "__main__":
    main()
