from __future__ import annotations

from enum import StrEnum


class Augmentation(StrEnum):
    """All supported data augmentations"""

    ROTATE = "rotate"
    NOISE = "noise"
    FLIP = "flip"  # vertical (top to bottom) flip
    MIRROR = "mirror"  # horizontal (left to right) flip
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SCALE = "scale"
    TRANSLATE = "translate"
    SHEAR = "shear"
    COLOR_JITTER = "color_jitter"
    SHARPNESS = "sharpness"

    @classmethod
    def operations_to_not_repeat(cls) -> list[Augmentation]:
        """Returns a list of augmentations that do not make sense to repeat, such as FLIP or MIRROR."""
        return [cls.FLIP, cls.MIRROR]
