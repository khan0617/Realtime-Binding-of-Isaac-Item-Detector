import base64
import logging
import os
from dataclasses import dataclass
from threading import Event, Lock

import cv2
import numpy as np
from flask import Flask, Response, abort, render_template, send_from_directory, url_for
from flask_socketio import SocketIO  # type: ignore

from constants import (
    BBOX_COLOR,
    BBOX_TEXT_COLOR,
    CONF_THRESHOLD,
    DATA_DIR,
    ITEM_DIR,
    MODEL_WEIGHTS_100_EPOCHS_PATH,
    TARGET_BACKGROUND_SIZE,
    UNMODIFIED_FILE_NAME,
)
from image_processing.isaac_live_capture_handler import IsaacLiveCaptureHandler
from image_processing.screen_grabber import ScreenGrabber
from isaac_yolo.isaac_yolo_model import DetectionResult, IsaacYoloModel
from logging_config import configure_logging
from utils import get_isaac_item_from_yolo_class_id

configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class VisualizationSettings:
    """Visualization settings that can be updated from the front-end"""

    confidence_threshold: float
    bbox_color: str
    bbox_text_color: str


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

visualization_settings = VisualizationSettings(
    confidence_threshold=CONF_THRESHOLD, bbox_color=BBOX_COLOR, bbox_text_color=BBOX_TEXT_COLOR
)

# lock when we update the global_settings dict just in case
settings_lock = Lock()


@app.route("/")
def index() -> str:
    """Serve the main page of the app."""
    with settings_lock:
        settings = {
            "confidence_threshold": visualization_settings.confidence_threshold,
            "bbox_color": visualization_settings.bbox_color,
            "bbox_text_color": visualization_settings.bbox_text_color,
        }
    return render_template("index.html", visualization_settings=settings)


@app.route("/item_images/<path:item_id_tail>")
def item_images(item_id_tail: str) -> Response:
    """Serve item images from the local data/items directory.

    Ex: item_images/1 should return data/items/1/original_img.png

    Args:
        item_id_tail (str): Something like "1", where
        "1" corresponds to IsaacItem.get_image_id_tail() for an item.

    Returns:
        Response object for the file, or a 404 if unavailable.
    """
    project_root = os.path.dirname(app.instance_path)
    directory = os.path.join(project_root, DATA_DIR, ITEM_DIR, item_id_tail)
    logger.info("item_images(%s): Sending %s", item_id_tail, f"{directory}/{UNMODIFIED_FILE_NAME}")
    return send_from_directory(directory, UNMODIFIED_FILE_NAME)


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


@socketio.on("update_settings")
def handle_update_settings(data: dict[str, str | float]) -> None:
    """Handle incoming settings from the client.

    Example data object:

    {
        "confidenceThreshold": 0.6,
        "bboxColor": "#00FF00",
        "bboxLabelColor": "#000000"
    }

    Args:
        data (dict[str, str | float]): Dict containing bbox color (str), bbox text color (str), and conf threshold (float)
    """
    with settings_lock:
        if "confidenceThreshold" in data:
            visualization_settings.confidence_threshold = float(data["confidenceThreshold"])
            isaac_yolo_model.confidence_threshold = visualization_settings.confidence_threshold
        if "bboxColor" in data:
            visualization_settings.bbox_color = str(data["bboxColor"])
        if "bboxLabelColor" in data:
            visualization_settings.bbox_text_color = str(data["bboxLabelColor"])
        logger.info("handle_update_settings: Updated global settings: %s", str(visualization_settings))


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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode(".png", image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")  # type: ignore
        encoded_images.append(img_base64)

    return encoded_images


def get_detection_metadata(detection_results: list[list[DetectionResult]]) -> list[dict[str, str | float]]:
    """Reduce a 2d list of DetectionResult objects to the needed information to display on the frontend.

    Args:
        detection_results (list[list[DetectionResult]]): 2D list of detection results, obtained from calling
            IsaacLiveCaptureHandler.run_capture_and_inference().

    Returns:
        A list of dictionaries, one dict for each of the DetectionResult objects. Each dict is like:

        {
            "name": isaac_item.name,
            "img_url": f"/item_images/{isaac_item.get_image_id_tail()}",
            "wiki_url": isaac_item.wiki_url,
            "description": isaac_item.description,
            "confidence": single_result.confidence
        }
    """
    detected_item_metadata: list[dict[str, str | float]] = []
    for list_of_detection_results in detection_results:
        for single_result in list_of_detection_results:
            if single_result.confidence < visualization_settings.confidence_threshold:
                continue
            isaac_item = get_isaac_item_from_yolo_class_id(str(single_result.class_id))
            logger.info("capture_and_emit: Got IsaacItem for %s", isaac_item.name)
            detected_item_metadata.append(
                {
                    "name": isaac_item.name,
                    "img_url": f"/item_images/{isaac_item.get_image_id_tail()}",
                    "wiki_url": isaac_item.wiki_url,
                    "description": isaac_item.description,
                    "confidence": single_result.confidence,
                }
            )

    return detected_item_metadata


def capture_and_emit() -> None:
    """Runs the capture and inference, then emits the results via SocketIO."""
    while should_run_capture.is_set():
        try:
            # convert each np.ndarray (image) to base64
            # we'll send the encoded images over via socketio.emit
            encoded_images = []

            # will also send the item metadata over so we can show them on screen.
            detected_item_metadata = []

            # run inference on the game window
            logger.info(
                "capture_and_emit: Calling run_capture_and_inference(bbox_color=%s, bbox_text_color=%s",
                visualization_settings.bbox_color,
                visualization_settings.bbox_text_color,
            )
            inference_results = isaac_live_capture_handler.run_capture_and_inference(
                bbox_color=visualization_settings.bbox_color, bbox_text_color=visualization_settings.bbox_text_color
            )

            if inference_results:
                all_detection_results = [dr for dr, _ in inference_results]
                detected_item_metadata = get_detection_metadata(all_detection_results)

                images_before_encoding = [img for _, img in inference_results]
                encoded_images = base64_encode_images(images_before_encoding, convert_bgr_to_rgb=False)

            else:
                # if we don't have any results we'll just take a screenshot.
                frame = screen_grabber.capture_window()
                encoded_images = base64_encode_images([frame], convert_bgr_to_rgb=True)
                logger.info("capture_and_emit: No items found on screen, sending screenshot.")

            # one last filter of "unicorn stump" or "lump of coal" from the metadata list since those are problematic
            detected_item_metadata = [
                d
                for d in detected_item_metadata
                if (not "unicorn stump" in d["name"].lower() and not "lump of coal" in d["name"].lower())  # type: ignore
            ]

            logger.info(
                "capture_and_emit: Emitting %d images and %d detected_item_metadata: %s",
                len(encoded_images),
                len(detected_item_metadata),
                f"detected_item_metadata: {({d['name']: d['confidence'] for d in detected_item_metadata})}",
            )
            socketio.emit("inference_update", {"images": encoded_images, "item_metadata": detected_item_metadata})

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("capture_and_emit: error during inference: %s", str(e))


def main() -> None:
    # default runs on http://localhost:5000
    logger.info("main: app.root_path = %s, app.instance_path = %s", app.root_path, app.instance_path)
    socketio.run(app)


if __name__ == "__main__":
    main()
