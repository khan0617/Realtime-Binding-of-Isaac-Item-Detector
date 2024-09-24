import pytest

from src.image_processing.bbox import CocoBbox, YoloBbox


@pytest.fixture
def img_width() -> int:
    return 1000


@pytest.fixture
def img_height() -> int:
    return 625


def test_coco_to_yolo_bbox(img_width: int, img_height: int) -> None:
    # gisen
    coco_bbox = CocoBbox(x=100, y=50, w=200, h=150)

    expected_x_center = (100 + 200 / 2) / img_width
    expected_y_center = (50 + 150 / 2) / img_height
    expected_width = 200 / img_width
    expected_height = 150 / img_height

    # when
    yolo_bbox = coco_bbox.to_yolo_bbox(img_width, img_height)

    # then
    assert yolo_bbox.x_center == pytest.approx(expected_x_center)
    assert yolo_bbox.y_center == pytest.approx(expected_y_center)
    assert yolo_bbox.width == pytest.approx(expected_width)
    assert yolo_bbox.height == pytest.approx(expected_height)


def test_yolo_to_coco_bbox(img_width: int, img_height: int) -> None:
    # given
    yolo_bbox = YoloBbox(x_center=0.25, y_center=0.16, width=0.2, height=0.24)

    expected_x = int((0.25 * img_width) - (0.2 * img_width / 2))
    expected_y = int((0.16 * img_height) - (0.24 * img_height / 2))
    expected_w = int(0.2 * img_width)
    expected_h = int(0.24 * img_height)

    # when
    coco_bbox = yolo_bbox.to_coco_bbox(img_width, img_height)

    # then
    assert coco_bbox.x == expected_x
    assert coco_bbox.y == expected_y
    assert coco_bbox.w == expected_w
    assert coco_bbox.h == expected_h
