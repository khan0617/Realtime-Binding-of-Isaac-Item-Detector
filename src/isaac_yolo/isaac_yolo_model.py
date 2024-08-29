from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO  # type: ignore
from ultralytics.engine.results import Boxes, Results  # type: ignore

from constants import MODEL_WEIGHTS_100_EPOCHS_PATH, TARGET_BACKGROUND_SIZE
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """
    Represents a single detection result from a YOLO model.

    Use this class to represent the dicts returned by `result.summary()`.
    """

    name: str
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectionResult:
        """
        Create a DetectionResult object from a dictionary.

        Args:
            d (dict): A dictionary with keys 'name', 'class', 'confidence', and 'box'.
                                   Example:
                                   {'box': {'x1': 404.6, 'x2': 481.8, 'y1': 334.9, 'y2': 412.1},
                                    'class': 317,
                                    'confidence': 0.88475,
                                    'name': "Mom's Coin Purse"}

        Returns:
            DetectionResult: An instance of DetectionResult populated with the data from the dictionary.
        """
        return cls(
            name=d["name"],
            class_id=d["class"],
            confidence=d["confidence"],
            x1=d["box"]["x1"],
            y1=d["box"]["y1"],
            x2=d["box"]["x2"],
            y2=d["box"]["y2"],
        )


class IsaacYoloModel:
    """IsaacYoloModel is responsible for making bounding-box predictions on images or video of the game."""

    def __init__(self, path_to_weights, img_size: tuple[int, int], confidence_threshold: float = 0.2) -> None:
        """
        Initialize the IsaacYoloModel.

        Args:
            path_to_weights (str): Path to the trained YOLO model weights.
            img_size (tuple[int, int]): Image size
            confidence_threshold (float): The minimum confidence to consider an image as "detected" and plottable.
        """
        self._img_size = img_size
        self._confidence_threshold = confidence_threshold
        self._model = self._load_model(path_to_weights)

    def _load_model(self, path_to_weights: str, task: str = "detect") -> YOLO:
        """
        Load a YOLO model for the specified task.

        Args:
            path_to_weights (str): Where the model weights are stored, ex: 'model_weights/isaac_model.pt'
            task (str):
        """
        model = YOLO(path_to_weights, task=task)
        logger.info("Model loaded from path %s", path_to_weights)
        return model

    def predict(self, image_paths: list[str], stream: bool = False) -> Iterable[Results]:
        """
        Perform object detection using the YOLO model on the images supplied by image_paths.

        Args:
            image_paths (str): List of paths to images to run inference on.
                To run inference on one image, pass a list with 1 element like: ['screenshots/img1.jpg'].
            stream (bool): When True, run the YOLO model in stream mode, which returns a generator of Result objects.
                When False, return a list of Results instead, loaded into memory all at once.

        Returns:
            List[Results] when stream=False, otherwise Generator[Results] when Stream=True.
        """
        results: Iterable[Results] = self._model.predict(image_paths, stream=stream)
        return results

    # pylint: disable=all
    def visulize_results(
        self, results: Results | Iterable[Results], show: bool = True, save_path: str | None = None
    ) -> None:
        """
        Viusualize prediction results on the images.

        Args:
            results (Results | Iterable[Results]): Results object or an Iterable of Results objects from the YOLO model prediction.
            show (bool): If True, display the image using matplotlib.
            save_path (str | None): If specified, save the image with bounding boxes to this path.
        """
        if not isinstance(results, Iterable):
            results = [results]

        for result in results:
            copy_of_orig_img = cast(np.ndarray, result.orig_img).copy()
            detection_results = [DetectionResult.from_dict(d) for d in result.summary()]

            if not detection_results:
                logger.info("No detection_results for image %s", result.path)
                continue

            for det in detection_results:
                # draw the bounding box, modifies the image in place
                cv2.rectangle(copy_of_orig_img, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), (0, 255, 0), 2)

                # prepare the label
                label = f"{det.name}: {det.confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_y = max(int(det.y1), label_size[1] + 10)

                # draw a rectangle behind the label
                cv2.rectangle(
                    copy_of_orig_img,
                    (int(det.x1), label_y - label_size[1] - 10),
                    (int(det.x1) + label_size[0], label_y + 5),
                    (0, 255, 0),
                    cv2.FILLED,
                )

                # put the label text on the image
                cv2.putText(
                    copy_of_orig_img, label, (int(det.x1), label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                )

            if show:
                # cv2 stores images differently than plt: https://stackoverflow.com/questions/38598118/difference-between-plt-imshow-and-cv2-imshow
                plt.imshow(cv2.cvtColor(copy_of_orig_img, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.show()

            if save_path is not None:
                cv2.imwrite(save_path, copy_of_orig_img)
                logger.info("Image with detections saved to %s", save_path)


def main():
    """Example usage of IsaacYoloModel."""
    isaac_yolo_model = IsaacYoloModel(path_to_weights=MODEL_WEIGHTS_100_EPOCHS_PATH, img_size=TARGET_BACKGROUND_SIZE)

    # download some images you find online for this to work.
    # since these are in my downloads folder which is not in this repo.
    image_paths = [
        Path(r"C:\Users\hamza\Downloads\angel_room.webp"),
        Path(r"C:\Users\hamza\Downloads\death_certificate.jpg"),
        Path(r"C:\Users\hamza\Downloads\devil_room.jpg"),
        Path(r"C:\Users\hamza\Downloads\devil_room_2.jpg"),
        Path(r"C:\Users\hamza\Downloads\devil_room_3.webp"),
    ]

    results = isaac_yolo_model.predict(image_paths=image_paths)
    isaac_yolo_model.visulize_results(results, show=True)
    print(f"{results = }")


if __name__ == "__main__":
    main()
