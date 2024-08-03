from dataclasses import dataclass


@dataclass(frozen=True)
class CocoBbox:
    """Represents a bounding box in an image, in COCO format.

    (x, y) is the top left point, and (x+w, y+h) is the bottom right of the bounding box.
    https://cocodataset.org/#home
    https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco

    """

    x: int
    y: int
    w: int
    h: int

    def to_yolo_bbox(self, img_width: int, img_height: int) -> "YoloBbox":
        """Converts this COCO bounding box to YOLO format.

        Args:
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            CocoBbox: Bounding box in COCO format.
        """
        x_center = (self.x + self.w / 2) / img_width
        y_center = (self.y + self.h / 2) / img_height
        width = self.w / img_width
        height = self.h / img_height
        return YoloBbox(x_center, y_center, width, height)


@dataclass(frozen=True)
class YoloBbox:
    """Represents a bounding box in an image, in YOLO format.

    (x_center, y_center) represent the center of the bounding box and (w, h) are the dimensions of the box.
    The values are normalized, meaning if your boxes are in "pixels", x_center and width are divided
    by image width, and y_center and height are divided by image height.

    See:
    https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
    https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#yolo

    """

    x_center: float
    y_center: float
    width: float
    height: float

    def to_coco_bbox(self, img_width: int, img_height: int) -> CocoBbox:
        """Converts the YOLO bounding box to COCO format.

        Ex: For a (1000, 625) PIL image pass in img_width=1000, img_height=625.

        Args:
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            CocoBbox: Bounding box in COCO format.
        """
        x = (self.x_center * img_width) - (self.width * img_width / 2)
        y = (self.y_center * img_height) - (self.height * img_height / 2)
        w = self.width * img_width
        h = self.height * img_height
        return CocoBbox(int(x), int(y), int(w), int(h))
