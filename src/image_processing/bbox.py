from dataclasses import dataclass


@dataclass(frozen=True)
class Bbox:
    """Represents a bounding box in an image, in COCO format.

    (x, y) is the top left point, and (x+w, y+h) is the bottom right of the bounding box.
    https://cocodataset.org/#home
    https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco

    """

    x: int
    y: int
    w: int
    h: int
