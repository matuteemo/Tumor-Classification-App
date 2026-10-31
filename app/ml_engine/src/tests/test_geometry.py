import numpy as np
import pytest

from src.preprocessing.geometry import SquareBox, crop_square, resize_square, square_box


def extent(mask: np.ndarray) -> tuple[int, int]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1


def test_square_box_is_centered_on_the_region_with_the_margin() -> None:
    region = np.zeros((200, 300), dtype=bool)
    region[60:100, 100:180] = True

    box = square_box(region, margin=0.05)

    assert box == SquareBox(top=36, left=96, side=88)


def test_square_box_without_margin_is_the_longer_side() -> None:
    region = np.zeros((200, 300), dtype=bool)
    region[60:100, 100:180] = True

    assert square_box(region, margin=0.0).side == 80


def test_full_image_region_gives_a_square_centered_on_the_image() -> None:
    box = square_box(np.ones((300, 400), dtype=bool), margin=0.05)

    assert box == SquareBox(top=-70, left=-20, side=440)


def test_crop_inside_the_image_is_the_slice() -> None:
    image = np.random.default_rng(0).random((50, 60)).astype(np.float32)

    square = crop_square(image, SquareBox(top=5, left=10, side=30))

    assert square.dtype == np.float32
    assert np.array_equal(square, image[5:35, 10:40])


def test_crop_past_the_corner_is_black_outside_the_image() -> None:
    image = np.ones((50, 60), dtype=np.float32)

    square = crop_square(image, SquareBox(top=-4, left=-6, side=20))

    assert square.shape == (20, 20)
    assert not square[:4].any() and not square[:, :6].any()
    assert square[4:, 6:].all()


def test_crop_past_the_far_borders_is_black() -> None:
    image = np.ones((50, 60), dtype=np.float32)

    square = crop_square(image, SquareBox(top=40, left=50, side=20))

    assert square[:10, :10].all()
    assert not square[10:].any() and not square[:, 10:].any()


def test_crop_keeps_pixels_outside_the_region() -> None:
    image = np.full((100, 100), 0.3, dtype=np.float32)
    region = np.zeros((100, 100), dtype=bool)
    region[40:60, 40:60] = True
    image[region] = 0.9

    square = crop_square(image, square_box(region, margin=0.25))

    assert set(np.unique(square)) == {np.float32(0.3), np.float32(0.9)}


@pytest.mark.parametrize("side", [512, 300, 88])
def test_resize_gives_the_requested_size(side: int) -> None:
    square = np.random.default_rng(1).random((side, side)).astype(np.float32)

    assert resize_square(square, 224).shape == (224, 224)


def test_crop_and_resize_keep_the_aspect_ratio() -> None:
    image = np.zeros((300, 400), dtype=np.float32)
    image[130:170, 160:240] = 1.0

    square = crop_square(image, square_box(image > 0, margin=0.05))
    output = resize_square(square, 224) > 0.5

    height, width = extent(output)
    assert abs(width - 2 * height) <= 2


def test_margin_rounds_half_up_and_side_is_an_int() -> None:
    region = np.zeros((100, 100), dtype=bool)
    region[10:60, 10:20] = True

    box = square_box(region, margin=0.05)

    assert box.side == 50 + 2 * 3
    assert all(type(value) is int for value in box)
