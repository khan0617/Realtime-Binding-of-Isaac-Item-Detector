import json
import logging
from functools import lru_cache

from constants import JSON_DUMP_FILE
from image_processing.bbox import YoloBbox
from logging_config import configure_logging
from scraping.isaac_item import IsaacItem

configure_logging()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _load_json_data(json_file: str = JSON_DUMP_FILE) -> dict[str, dict[str, str]]:
    """Load in the JSON data from JSON_DUMP_FILE. Contains data on Isaac Items.

    See scraper.py for more info on how objects are dumped to json.

    Args:
        json_file (str, optional): The json file where IsaacItem objects have been dumped.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data: dict[str, dict[str, str]] = json.load(f)
        return data


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

    Raises:
        ValueError on failure, meaning no json object with item_id_tail exists.
    """
    json_data = _load_json_data()
    if item_id_tail in json_data:
        return json_data[item_id_tail]["yolo_class_id"]
    raise ValueError(f"get_yolo_class_id_from_isaac_id_tail: No yolo_class_id exists for {item_id_tail = }.")


@lru_cache(maxsize=None)
def get_isaac_item_from_yolo_class_id(yolo_class_id: str) -> IsaacItem:
    """Get the IsaacItem object corresponding to this yolo_class_id.

    Example: The "Forget Me Now" item has item_id="5.100.127" and yolo_class_id="24".
    get_isaac_item_from__item_id_tail("24") will return IsaacItem(name="Forget Me Now", item_id="5.100.127", ...)

    Args:
        yolo_class_id (str): The yolo class id for an IsaacItem.

    Returns:
        IsaacItem populated with the appropriate information.

    Raises:
        ValueError: If no object with the yolo_class_id is found in the json data.
    """
    json_data = _load_json_data()
    for _, item_data in json_data.items():
        if yolo_class_id == item_data["yolo_class_id"]:
            return IsaacItem.from_dict(item_data)
    raise ValueError(f"get_isaac_item_from_yolo_class_id: No yolo_class_id exists for {yolo_class_id = }.")


@lru_cache(maxsize=None)
def get_id_name_mapping() -> dict[int, str]:
    """Return the YOLO class_id: class_name mapping required for the YAML config.
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


@lru_cache(maxsize=None)
def hex_to_bgr(hex_color: str) -> tuple[int, ...]:
    """Convers a hex color string to a BGR tuple for OpenCV.

    Args:
        hex_color (str): Hex color string (ex: "#FF0000")

    Returns:
        tuple[int, int, int]: BGR tuple (ex: (0, 0, 255)).
        Tuple values will be in range [0, 255]
    """
    hex_color = hex_color.lstrip("#")
    rgb_tuple = tuple(int(hex_color[i : i + 2], base=16) for i in (0, 2, 4))
    bgr_tuple = rgb_tuple[::-1]  # reverse the rgb tuple to get bgr
    return bgr_tuple


def main():
    # TODO: remove these asserts and put them in test/
    sad_onion_id_tail = "1"
    assert get_yolo_class_id_from_item_id_tail(sad_onion_id_tail) == "419"

    forget_me_now_yolo_class_id = "24"
    assert get_isaac_item_from_yolo_class_id(forget_me_now_yolo_class_id).name == "Forget Me Now"

    black_hex = "#000000"
    black_bgr = hex_to_bgr(black_hex)
    assert black_bgr == (0, 0, 0)

    green_hex = "#00FF00"
    green_bgr = hex_to_bgr(green_hex)
    assert green_bgr == (0, 255, 0)

    print(get_isaac_item_from_yolo_class_id("643"))  # should be Guppy's Eye.


if __name__ == "__main__":
    main()
