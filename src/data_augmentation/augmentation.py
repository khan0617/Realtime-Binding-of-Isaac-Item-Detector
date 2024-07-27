from enum import StrEnum


class Augmentation(StrEnum):
    """All supported data augmentations"""

    ROTATE = "rotate"
    NOISE = "noise"
    FLIP = "flip"
    MIRROR = "mirror"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SCALE = "scale"
    TRANSLATE = "translate"
    SHEAR = "shear"
    COLOR_JITTER = "color_jitter"
