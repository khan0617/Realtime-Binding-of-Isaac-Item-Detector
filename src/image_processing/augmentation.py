from __future__ import annotations

from enum import StrEnum


class Augmentation(StrEnum):
    """All supported data augmentations"""

    ROTATE = "rotate"
    NOISE = "noise"
    VERTICAL_FLIP = "flip"  # vertical (top to bottom) flip
    HORIZONTAL_MIRROR = "mirror"  # horizontal (left to right) flip
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SCALE = "scale"
    TRANSLATE = "translate"
    SHEAR = "shear"
    COLOR_JITTER = "color_jitter"
    SHARPNESS = "sharpness"
    SMOOTH = "smooth"

    @classmethod
    def operations_to_not_repeat(cls) -> list[Augmentation]:
        """Returns a list of augmentations that do not make sense to repeat (those that are not randomized),
        such as FLIP or MIRROR."""
        return [cls.VERTICAL_FLIP, cls.HORIZONTAL_MIRROR, cls.SMOOTH]
