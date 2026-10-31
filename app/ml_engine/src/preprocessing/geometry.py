"""CNN image geometry of a 2D MRI slice (pipeline block B6): square crop around the brain."""

from __future__ import annotations

import math
from typing import NamedTuple

import cv2
import numpy as np

from src.preprocessing.brain_mask import bounding_box


class SquareBox(NamedTuple):
    """Square window in image coordinates; it may extend past the image borders."""

    top: int
    left: int
    side: int


def square_box(region: np.ndarray, margin: float) -> SquareBox:
    """Square centered on the bounding box of ``region``.

    Its side is the box's longer side plus ``margin`` of it at each end, rounded half up.
    ``region`` must have at least one true pixel (the B5 region of interest always does).
    """
    rows, cols = bounding_box(region)
    height, width = rows.stop - rows.start, cols.stop - cols.start
    longer = max(height, width)
    side = longer + 2 * math.floor(margin * longer + 0.5)
    top = rows.start + (height - side) // 2
    left = cols.start + (width - side) // 2
    return SquareBox(top, left, side)


def crop_square(image: np.ndarray, box: SquareBox) -> np.ndarray:
    """The pixels of ``image`` inside ``box``, black where the box lies outside the image."""
    height, width = image.shape
    top, left = max(box.top, 0), max(box.left, 0)
    bottom = min(box.top + box.side, height)
    right = min(box.left + box.side, width)
    padding = (
        (top - box.top, box.top + box.side - bottom),
        (left - box.left, box.left + box.side - right),
    )
    window = np.pad(image[top:bottom, left:right], padding)
    return window.astype(np.float32, copy=False)


def resize_square(square: np.ndarray, size: int) -> np.ndarray:
    """Resize a square image to ``size`` x ``size``.

    Area interpolation to shrink, linear to enlarge.
    """
    interpolation = cv2.INTER_AREA if square.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(square, (size, size), interpolation=interpolation)
