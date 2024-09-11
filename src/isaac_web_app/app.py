import base64
import logging
import time
from threading import Event

import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO  # type: ignore

from constants import MODEL_WEIGHTS_100_EPOCHS_PATH, TARGET_BACKGROUND_SIZE
from image_processing.isaac_live_capture_handler import IsaacLiveCaptureHandler
from image_processing.screen_grabber import ScreenGrabber
from isaac_yolo.isaac_yolo_model import IsaacYoloModel
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app)

screen_grabber = ScreenGrabber()
isaac_yolo_model = IsaacYoloModel(
    path_to_weights=MODEL_WEIGHTS_100_EPOCHS_PATH,
    img_size=TARGET_BACKGROUND_SIZE,
)
isaac_live_capture_handler = IsaacLiveCaptureHandler(screen_grabber, isaac_yolo_model)

# event to signal when to stop the background task
should_run_capture = Event()


@app.route("/")
def index() -> str:
    """Serve the main page of the app."""
    return render_template("index.html")


@socketio.on("connect")
def on_connect():
    """Handle client connection and start the background capture task."""
    logger.info("on_connect: Client connected, starting capture task.")
    should_run_capture.set()
    socketio.start_background_task(target=capture_and_emit)


@socketio.on("disconnect")
def on_disconnect():
    """Handle client disconnection and stop the background capture task."""
    logger.info("on_disconnect: Client disconnectd, stopping capture task.")
    should_run_capture.clear()


# pylint: disable=no-member
def base64_encode_images(images: list[np.ndarray], convert_bgr_to_rgb: bool = False) -> list[str]:
    """
    Convert each np.ndarray to base64.

    Args:
        images (list[np.ndarray]): List of images to encode. Should be yolo model inference output.
        convert_bgr_to_rgb (bool: optional): If True, convert the images from bgr to rgb format before encoding.

    Returns:
        List of images encoded as base64 strings.
    """
    encoded_images = []
    for image in images:
        if convert_bgr_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB if needed
        _, buffer = cv2.imencode(".png", image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        encoded_images.append(img_base64)

    return encoded_images


def capture_and_emit():
    """Runs the capture and inference, then emits the results via SocketIO."""
    while should_run_capture.is_set():
        try:
            results = isaac_live_capture_handler.run_capture_and_inference()
            # convert each np.ndarray (image) to base64
            encoded_images = []

            if results:
                encoded_images = base64_encode_images(results, convert_bgr_to_rgb=False)
                logger.info("capture_and_emit: Encoded %d inference results", len(encoded_images))

            else:
                # if we don't have any results we'll just take a screenshot.
                frame = screen_grabber.capture_window()
                encoded_images = base64_encode_images([frame], convert_bgr_to_rgb=True)
                logger.info("capture_and_emit: No items found on screen, sending screenshot.")

            socketio.emit("inference_update", {"images": encoded_images})

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("capture_and_emit: error during inference: %s", str(e))


def main():
    # default runs on http://localhost:5000
    socketio.run(app)


if __name__ == "__main__":
    main()
