import json
import logging
import os

from constants import JSON_DUMP_FILE
from image_processing.bbox import YoloBbox
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

_JSON_DATA: dict[str, dict[str, str]] = {}


def _load_json_data() -> None:
    global _JSON_DATA
    if not _JSON_DATA:
        with open(JSON_DUMP_FILE, "r", encoding="utf-8") as f:
            _JSON_DATA = json.load(f)


def convert_item_name_to_id(item_name: str) -> int:
    """Convert this item name (url-encoded) to its int ID.

    Relies on the JSON dumpfile from scraper.py existing.
    For example, item_name == "%3F%3F%3F%27s_Only_Friend" has an associated "item_id": "5.100.320"
    in the json. We'll return 320 as an int.

    """
    assert os.path.exists(JSON_DUMP_FILE), f"{JSON_DUMP_FILE} must exist to convert name to ID."
    _load_json_data()
    item_id: str = _JSON_DATA[item_name]["item_id"]  # like "5.100.320"
    return int(item_id.split(".")[-1])


def read_yolo_label_file(filepath: str) -> tuple[int, YoloBbox]:
    """Read the specified YOLO label file and return the class ID and bounding box.

    Each line in the file is formatted as: `<class_id> <x_center> <y_center> <width> <height>`

    Args:
        filepath (str): Path to the YOLO label file.

    Returns:
        tuple[int, YoloBbox]: Class ID and YoloBbox instance.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        parts = line.split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        yolo_bbox = YoloBbox(x_center, y_center, width, height)
        return class_id, yolo_bbox
