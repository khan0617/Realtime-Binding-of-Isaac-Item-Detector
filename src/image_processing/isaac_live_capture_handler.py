import logging
import time

import numpy as np
from PIL import Image

from constants import MODEL_WEIGHTS_100_EPOCHS_PATH, TARGET_BACKGROUND_SIZE
from image_processing.screen_grabber import ScreenGrabber
from isaac_yolo.isaac_yolo_model import IsaacYoloModel
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class IsaacLiveCaptureHandler:
    """
    Class to handle live capturing of the Binding of Isaac game window
    and running the images through YOLO for inference.
    """

    def __init__(self, screen_grabber: ScreenGrabber, isaac_yolo_model: IsaacYoloModel) -> None:
        """
        Initialize the IsaacLiveCaptureHandler.

        Args:
            screen_grabber (ScreenGrabber): The screen capture utility.
            isaac_yolo_model (IsaacYoloModel): The YOLO model for object detection.
        """
        self._screen_grabber = screen_grabber
        self._isaac_yolo_model = isaac_yolo_model

    def run_capture_and_inference(self, show: bool = False) -> list[np.ndarray] | None:
        """
        Capture the game window and run inference.

        Args:
            show (bool, optional): If True, show each of the detection results with bounding boxes and labels draw.

        Returns:
            list[np.ndarray] | None: A list of images with bounding boxes + labels overlaid.
                If no results were available or the call failed, return None.
        """
        try:
            frame = Image.fromarray(self._screen_grabber.capture_window())
            results = self._isaac_yolo_model.predict([frame], stream=False)
            visualized_results = self._isaac_yolo_model.visulize_results(
                results=results, show=show, return_visualized_results=True, skip_unicorn_stump_and_coal=True
            )
            return visualized_results

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("run_capture_and_inference: Error during capture or inference: %s", str(e))
            return None


def main():
    # Initialize the components
    screen_grabber = ScreenGrabber()
    isaac_yolo_model = IsaacYoloModel(
        path_to_weights=MODEL_WEIGHTS_100_EPOCHS_PATH,
        img_size=TARGET_BACKGROUND_SIZE,
    )
    isaac_live_capture_handler = IsaacLiveCaptureHandler(screen_grabber, isaac_yolo_model)

    # let's run every 5 seconds, up to 30 seconds, and plot the results.
    try:
        start_time = time.time()
        while time.time() - start_time <= 30:
            isaac_live_capture_handler.run_capture_and_inference(show=True)
            time.sleep(5)
    finally:
        logger.info("Stopped live capture and inference.")


if __name__ == "__main__":
    main()
