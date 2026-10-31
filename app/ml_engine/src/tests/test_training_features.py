from pathlib import Path

import numpy as np

from src.classical.config import RadiomicsConfig
from src.contracts import QualityMetrics, TumorLabel
from src.errors import QualityCheckError, UnsupportedImageError
from src.preprocessing.prepare import PreprocessingConfig
from src.preprocessing.quality import QualityThresholds
from src.tests.synthetic import save, synthetic_slice
from src.training.dataset import Sample
from src.training.features import FeatureCache, config_fingerprint

RADIOMICS = RadiomicsConfig()


class CountingFeaturizer:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(path)
        if path.name.startswith("bad"):
            raise QualityCheckError("too noisy", QualityMetrics(1.0, 1.0, False))
        if path.name.startswith("odd"):
            raise UnsupportedImageError("not an image")
        value = float(len(path.name))
        return np.full(45, value, np.float32), np.full(8, value * 2, np.float32)


def make_samples(tmp_path: Path, names: list[str]) -> list[Sample]:
    samples = []
    for index, name in enumerate(names):
        path = save(synthetic_slice(index, size=64), tmp_path / name)
        samples.append(Sample(path, "Train", TumorLabel.GLIOMA, index))
    return samples


def cache(
    tmp_path: Path, featurize: CountingFeaturizer, fingerprint: str = "fp1"
) -> FeatureCache:
    return FeatureCache(tmp_path / "cache", fingerprint, featurize, RADIOMICS)


def test_table_rows_follow_the_samples(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png", "bbbb.png"])

    table, skipped = cache(tmp_path, CountingFeaturizer()).table(samples)

    assert skipped == []
    assert table.samples == tuple(samples)
    assert table.radiomics.shape == (2, 45) and table.cnn.shape == (2, 8)
    assert table.matrix.shape == (2, 53)
    assert table.radiomics[0, 0] == len("a.png")
    assert len(table.radiomic_names) == 45


def test_second_run_reads_everything_from_the_cache(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png", "b.png"])
    featurizer = CountingFeaturizer()
    first, _ = cache(tmp_path, featurizer).table(samples)

    second, _ = cache(tmp_path, featurizer).table(samples)

    assert len(featurizer.calls) == 2
    assert np.array_equal(first.matrix, second.matrix)


def test_only_new_images_are_computed(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png", "b.png", "c.png"])
    featurizer = CountingFeaturizer()
    cache(tmp_path, featurizer).table(samples[:2])

    cache(tmp_path, featurizer).table(samples)

    assert [p.name for p in featurizer.calls] == ["a.png", "b.png", "c.png"]


def test_changing_an_image_recomputes_it(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png"])
    featurizer = CountingFeaturizer()
    cache(tmp_path, featurizer).table(samples)
    save(synthetic_slice(99, size=64), samples[0].path)

    cache(tmp_path, featurizer).table(samples)

    assert len(featurizer.calls) == 2


def test_a_new_fingerprint_does_not_reuse_the_old_cache(tmp_path: Path) -> None:
    samples = make_samples(tmp_path, ["a.png"])
    featurizer = CountingFeaturizer()
    cache(tmp_path, featurizer, "fp1").table(samples)

    cache(tmp_path, featurizer, "fp2").table(samples)

    assert len(featurizer.calls) == 2


def test_rejected_images_are_skipped_with_their_reason_and_cached(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path, ["a.png", "bad.png", "odd.png"])
    featurizer = CountingFeaturizer()

    table, skipped = cache(tmp_path, featurizer).table(samples)
    cache(tmp_path, featurizer).table(samples)

    assert [s.path.name for s in table.samples] == ["a.png"]
    assert {s.path.name: s.reason for s in skipped} == {
        "bad.png": "too noisy",
        "odd.png": "not an image",
    }
    assert len(featurizer.calls) == 3


def test_empty_input_gives_an_empty_table(tmp_path: Path) -> None:
    table, skipped = cache(tmp_path, CountingFeaturizer()).table([])

    assert table.matrix.shape[0] == 0 and skipped == []


def test_fingerprint_changes_with_any_input() -> None:
    base = config_fingerprint(PreprocessingConfig(), RADIOMICS, "sha")

    assert base == config_fingerprint(PreprocessingConfig(), RADIOMICS, "sha")
    assert base != config_fingerprint(
        PreprocessingConfig(cnn_size=256), RADIOMICS, "sha"
    )
    assert base != config_fingerprint(
        PreprocessingConfig(), RadiomicsConfig(lbp_points=16), "sha"
    )
    assert base != config_fingerprint(PreprocessingConfig(), RADIOMICS, "other")
    assert base != config_fingerprint(
        PreprocessingConfig(min_mask_fraction=0.1), RADIOMICS, "sha"
    )
    assert base != config_fingerprint(
        PreprocessingConfig(crop_margin=0.1), RADIOMICS, "sha"
    )
    assert base != config_fingerprint(
        PreprocessingConfig(thresholds=QualityThresholds(metrics_version=1)),
        RADIOMICS,
        "sha",
    )
