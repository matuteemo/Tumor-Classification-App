import numpy as np

from src.preprocessing.brain_mask import bounding_box, foreground_mask, region_of_interest
from src.tests.synthetic import head_phantom


def disc(size: int, radius: float, center: tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return np.hypot(yy - center[0], xx - center[1]) < radius


def test_foreground_mask_covers_the_head_and_not_the_background() -> None:
    phantom = head_phantom(256)

    mask = foreground_mask(phantom)

    assert mask[128, 128]
    assert not mask[0, 0] and not mask[-1, -1]
    assert mask.dtype == bool and mask.shape == phantom.shape


def test_foreground_mask_of_flat_images() -> None:
    assert foreground_mask(np.full((32, 32), 0.5, dtype=np.float32)).all()
    assert not foreground_mask(np.zeros((32, 32), dtype=np.float32)).any()


def test_foreground_mask_keeps_only_the_largest_region() -> None:
    gray = np.zeros((256, 256), dtype=np.float32)
    gray[disc(256, 60, (90, 90))] = 0.8
    gray[disc(256, 20, (210, 210))] = 0.8

    mask = foreground_mask(gray)

    assert mask[90, 90]
    assert not mask[210, 210]


def test_foreground_mask_fills_enclosed_dark_regions() -> None:
    gray = np.zeros((256, 256), dtype=np.float32)
    gray[disc(256, 90, (128, 128))] = 0.8
    gray[disc(256, 20, (128, 128))] = 0.0

    assert foreground_mask(gray)[128, 128]


def test_region_is_the_mask_when_it_is_large_enough() -> None:
    mask = foreground_mask(head_phantom(256))

    region = region_of_interest(mask, 0.05)

    assert np.array_equal(region, mask)


def test_empty_mask_falls_back_to_the_full_image() -> None:
    mask = foreground_mask(np.zeros((64, 64), dtype=np.float32))

    assert region_of_interest(mask, 0.05).all()


def test_tiny_bright_spot_falls_back_to_the_full_image() -> None:
    gray = np.zeros((256, 256), dtype=np.float32)
    gray[disc(256, 10, (128, 128))] = 1.0
    mask = foreground_mask(gray)

    region = region_of_interest(mask, 0.05)

    assert mask.mean() < 0.05
    assert region.all() and region.dtype == bool


def test_empty_mask_falls_back_even_with_a_zero_fraction() -> None:
    assert region_of_interest(np.zeros((8, 8), dtype=bool), 0.0).all()


def test_bounding_box_is_the_smallest_window_with_every_true_pixel() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[5, 8] = mask[9, 19] = True

    assert bounding_box(mask) == (slice(5, 10), slice(8, 20))
