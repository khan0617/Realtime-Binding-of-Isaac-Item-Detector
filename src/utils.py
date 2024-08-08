import json
import logging
import os
from functools import lru_cache

from constants import JSON_DUMP_FILE
from image_processing.bbox import YoloBbox
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _load_json_data() -> dict[str, dict[str, str]]:
    """Load in the JSON data from JSON_DUMP_FILE. Contains data on Isaac Items.
    See scraper.py for more info on how objects are dumped to json.
    """
    with open(JSON_DUMP_FILE, "r", encoding="utf-8") as f:
        data: dict[str, dict[str, str]] = json.load(f)
        return data


def convert_item_name_to_id(item_name: str) -> int:
    """Convert this item name (url-encoded) to its int ID.

    Relies on the JSON dumpfile from scraper.py existing.
    For example, item_name == "%3F%3F%3F%27s_Only_Friend" has an associated "item_id": "5.100.320"
    in the json. We'll return 320 as an int.
    """
    assert os.path.exists(JSON_DUMP_FILE), f"{JSON_DUMP_FILE} must exist to convert name to ID."
    json_data = _load_json_data()
    item_id: str = json_data[item_name]["item_id"]  # like "5.100.320"
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


@lru_cache(maxsize=None)
def get_yolo_class_id_from_item_id_tail(item_id_tail: str) -> str:
    """Convert the IsaacItem id tail to its yolo class id.

    Example: Sad Onion is item "1", but its yolo_class_id is "419".
    So get_yolo_class_id_from_isaac_id_tail("1") returns "419".

    Args:
        item_id_tail (str): The end of an IsaacItem item_id. Ex. If item_id is "5.100.105", tail is "105".

    Returns:
        The yolo_class_id which represents this IsaacItem.
        Raise ValueError on failure.
    """
    json_data = _load_json_data()
    for tail, item_data in json_data.items():
        if item_id_tail == tail:
            return item_data["yolo_class_id"]
    raise ValueError(f"get_yolo_class_id_from_isaac_id_tail: No yolo_class_id exists for {item_id_tail = }.")


@lru_cache(maxsize=None)
def get_id_name_mapping() -> dict[int, str]:
    """Return the YOLO class_id: class_name mapping required for the YAML config.
    If this function is called multiple times, the map will only be
    Ex. {0: 'person', 1: 'car'} could correspond to this in yaml:

    names:
        0: person
        1: car
    """
    json_data = _load_json_data()
    id_name_map = {}
    for _, item_data in json_data.items():
        id_name_map[int(item_data["yolo_class_id"])] = item_data["name"]
    return id_name_map


def main():
    sad_onion_id_tail = "1"
    assert get_yolo_class_id_from_item_id_tail(sad_onion_id_tail) == "419"


if __name__ == "__main__":
    main()
