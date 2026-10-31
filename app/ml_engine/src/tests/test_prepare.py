import logging
from pathlib import Path

import numpy as np
import pytest

from src.errors import QualityCheckError
from src.preprocessing.brain_mask import foreground_mask
from src.preprocessing.loading import decode_mri_image, load_mri_image
from src.preprocessing.prepare import PreprocessingConfig, prepare_image
from src.preprocessing.quality import QualityThresholds, measure_quality
from src.tests.synthetic import head_phantom, save, synthetic_slice

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def slice_array(seed: int = 1, size: int = 256) -> np.ndarray:
    return np.asarray(synthetic_slice(seed, size), dtype=np.float32) / 255.0


def test_outputs_have_the_expected_shape_and_type() -> None:
    result = prepare_image(slice_array())

    assert result.radiomics_image.shape == (256, 256)
    assert result.radiomics_image.dtype == np.float32
    assert result.cnn_input.shape == (3, 224, 224)
    assert result.cnn_input.dtype == np.float32


def test_mask_is_the_brain_mask_of_the_image() -> None:
    gray = slice_array()

    result = prepare_image(gray)

    assert result.mask.shape == (256, 256) and result.mask.dtype == bool
    assert np.array_equal(result.mask, foreground_mask(gray))
    assert result.mask[128, 128] and not result.mask[0, 0]


def test_small_mask_falls_back_but_quality_uses_the_mask(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gray = head_phantom()
    raw = foreground_mask(gray)
    config = PreprocessingConfig(
        min_mask_fraction=raw.mean() + 0.01, thresholds=QualityThresholds(0, 0)
    )

    with caplog.at_level(logging.WARNING, logger="src.preprocessing.prepare"):
        result = prepare_image(gray, config)

    assert result.mask.all()
    assert "using the full image" in caplog.text
    expected = measure_quality(gray, mask=raw)
    assert (result.quality.snr, result.quality.entropy) == (
        expected.snr,
        expected.entropy,
    )


def test_training_and_inference_paths_give_the_same_region(tmp_path: Path) -> None:
    path = save(synthetic_slice(3, size=200), tmp_path / "slice.png")

    training = prepare_image(load_mri_image(path))
    inference = prepare_image(decode_mri_image(path.read_bytes()))

    assert np.array_equal(training.mask, inference.mask)
    assert np.array_equal(training.radiomics_image, inference.radiomics_image)
    assert np.array_equal(training.cnn_input, inference.cnn_input)


def test_cnn_input_is_normalized_per_channel() -> None:
    config = PreprocessingConfig(
        crop_margin=0.0,
        cnn_mean=(0.5, 0.5, 0.5),
        cnn_std=(0.25, 0.25, 0.25),
        thresholds=QualityThresholds(0, 0),
    )
    gray = 0.4 + 0.5 * np.random.default_rng(2).random((224, 224), dtype=np.float32)

    result = prepare_image(gray, config)

    assert result.mask.all()
    channels = result.cnn_input
    assert np.allclose(channels[0], channels[1]) and np.allclose(
        channels[1], channels[2]
    )
    assert np.allclose(channels[0], (gray - 0.5) / 0.25, atol=1e-5)


@pytest.mark.parametrize("margin", [-0.1, 1.5])
def test_crop_margin_outside_zero_to_one_is_refused(margin: float) -> None:
    with pytest.raises(ValueError, match="crop_margin"):
        PreprocessingConfig(crop_margin=margin)


def test_cnn_size_is_configurable_and_small_images_are_upscaled() -> None:
    result = prepare_image(slice_array(size=100), PreprocessingConfig(cnn_size=128))

    assert result.cnn_input.shape == (3, 128, 128)


@pytest.mark.parametrize("shape", [(512, 512), (300, 400), (200, 630)])
def test_cnn_input_has_a_fixed_size_for_any_source_shape(
    shape: tuple[int, int],
) -> None:
    rng = np.random.default_rng(4)
    gray = np.zeros(shape, dtype=np.float32)
    height, width = shape
    head = gray[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    head[:] = 0.4 + 0.5 * rng.random(head.shape)

    result = prepare_image(gray, PreprocessingConfig(thresholds=QualityThresholds(0, 0)))

    assert result.cnn_input.shape == (3, 224, 224)
    assert result.radiomics_image.shape == shape


def test_small_head_on_a_large_frame_fills_the_cnn_input() -> None:
    frame = np.zeros((500, 500), dtype=np.float32)
    frame[150:350, 150:350] = np.asarray(synthetic_slice(5, size=200)) / 255.0

    result = prepare_image(frame, PreprocessingConfig(thresholds=QualityThresholds(0, 0)))

    raw = result.cnn_input[0] * 0.229 + 0.485
    rows = np.flatnonzero((raw > 0.05).any(axis=1))
    border = rows[0] / 224
    assert 0.03 < border < 0.12


def test_radiomics_image_is_the_loaded_image_and_not_stripped() -> None:
    gray = head_phantom()

    result = prepare_image(
        gray, PreprocessingConfig(thresholds=QualityThresholds(0, 0))
    )

    assert result.quality.skull_stripped is False
    assert np.array_equal(result.radiomics_image, gray)


def test_low_quality_image_raises_with_metrics() -> None:
    with pytest.raises(QualityCheckError) as error:
        prepare_image(np.zeros((128, 128), dtype=np.float32))

    assert error.value.metrics.entropy == 0.0


def test_reported_quality_is_measured_on_the_loaded_image() -> None:
    gray = head_phantom()

    result = prepare_image(
        gray, PreprocessingConfig(thresholds=QualityThresholds(0, 0))
    )

    original = measure_quality(gray)
    assert (result.quality.snr, result.quality.entropy) == (
        original.snr,
        original.entropy,
    )


@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="dataset not available in src/data")
def test_real_dataset_sample_passes() -> None:
    sample = next(path for path in sorted(DATA_DIR.rglob("*.jpg")))

    prepare_image(load_mri_image(sample))
