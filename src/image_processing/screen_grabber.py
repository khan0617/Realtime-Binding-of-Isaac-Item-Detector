"""
`screen_grabber.py`

Utilities to find and capture the Isaac window on screen.
Works for Windows operating systems only.
"""

import logging
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygetwindow as gw  # type: ignore
from mss.windows import MSS as mss
from PIL import Image
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
        self._mss = mss()

    def get_isaac_window(self) -> Win32Window | None:
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

        logger.warning("get_isaac_window: Failed to find Binding of Isaac Window! Are you running the game?")
        return None

    def capture_window(self, window: Win32Window) -> np.ndarray:
        """
        Capture the specific window.
        To capture properly, the window must be in the foreground.

        Args:
            window (Win32Window): The window object to capture.

        Returns:
            The captured screen area has an np array.
        """
        with self._mss as sct:
            monitor = {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
            frame = np.array(sct.grab(monitor))
            color_corrected_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)  # pylint: disable=no-member
            return np.array(color_corrected_image)

    def display_captured_window(self, frame: np.ndarray) -> None:
        """
        Use matplotlib to show the captured frame.

        Args:
            frame (np.ndarray): The frame as an np array.
        """
        plt.imshow(frame)
        plt.axis("off")
        plt.title("Captured Isaac Window")
        plt.show()


def main():
    screen_grabber = ScreenGrabber()

    # get the window object for the Isaac game
    #   make sure you have the game running and it's in the foreground.
    window = screen_grabber.get_isaac_window()
    print(window)
    if window is None:
        sys.exit(1)

    # capture the Isaac window and visualize it
    frame = screen_grabber.capture_window(window)
    screen_grabber.display_captured_window(frame)


if __name__ == "__main__":
    main()
