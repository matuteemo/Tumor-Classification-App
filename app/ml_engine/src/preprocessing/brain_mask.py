"""Brain mask of a 2D MRI slice (pipeline block B5). No skull stripping."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu


def bright_regions(gray: np.ndarray) -> np.ndarray:
    """Every pixel above half the Otsu threshold, all components kept. ``gray`` must not be flat."""
    return gray > 0.5 * threshold_otsu(gray)


def foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Pixels clearly above the background: half the Otsu threshold, largest component, holes filled.

    This is the brain mask the quality gate measures on. A flat image is all
    foreground if it is not black, and empty otherwise.
    """
    if gray.max() <= gray.min():
        return np.full(gray.shape, gray.max() > 0, dtype=bool)
    foreground = bright_regions(gray)
    labels, count = ndimage.label(foreground)
    if count == 0:
        return foreground
    largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return ndimage.binary_fill_holes(labels == largest)


def region_of_interest(mask: np.ndarray, min_fraction: float) -> np.ndarray:
    """``mask``, or the full image when it is empty or covers less than ``min_fraction`` of it."""
    if not mask.any() or mask.mean() < min_fraction:
        return np.ones(mask.shape, dtype=bool)
    return mask


def bounding_box(mask: np.ndarray) -> tuple[slice, slice]:
    """Row and column slices of the smallest window holding every true pixel.

    ``mask`` must not be empty.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)
