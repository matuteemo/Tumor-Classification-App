import json
import shutil
from pathlib import Path

import pytest

from src.classical.config import RadiomicsConfig
from src.errors import ModelUnavailableError
from src.pipelines import BundleInferencePipeline
from src.pipelines.artifact import load_artifact
from src.preprocessing.prepare import PreprocessingConfig
from src.training.artifact import CLASSIFIER_FILE, MANIFEST_FILE, STATS_FILE
from src.training.run import TrainingResult


@pytest.fixture()
def copy(result: TrainingResult, tmp_path: Path) -> Path:
    return Path(shutil.copytree(result.artifact_dir, tmp_path / "artifact"))


def edit_manifest(directory: Path, **changes: object) -> None:
    path = directory / MANIFEST_FILE
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest))


def test_the_trained_configs_survive_the_manifest_round_trip(
    result: TrainingResult,
) -> None:
    artifact = load_artifact(result.artifact_dir)

    assert artifact.preprocessing == PreprocessingConfig()
    assert artifact.radiomics == RadiomicsConfig()
    assert artifact.version == "9.9.9"
    assert len(artifact.stats.names) == 45


def test_a_missing_folder_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="manifest not found"):
        BundleInferencePipeline.from_bundle(tmp_path / "nowhere")


def test_a_missing_file_is_unavailable(copy: Path) -> None:
    (copy / STATS_FILE).unlink()

    with pytest.raises(ModelUnavailableError, match="not found"):
        BundleInferencePipeline.from_bundle(copy)


@pytest.mark.parametrize(
    "name", [CLASSIFIER_FILE, STATS_FILE, "backbone.pt", "backbone.json"]
)
def test_an_altered_file_is_refused_before_anything_is_loaded(
    copy: Path, name: str
) -> None:
    with (copy / name).open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(ModelUnavailableError, match="sha256"):
        BundleInferencePipeline.from_bundle(copy)


def test_an_invalid_manifest_is_unavailable(copy: Path) -> None:
    (copy / MANIFEST_FILE).write_text("not json")

    with pytest.raises(ModelUnavailableError, match="invalid artifact manifest"):
        BundleInferencePipeline.from_bundle(copy)


def test_a_manifest_without_hashes_is_refused(copy: Path) -> None:
    edit_manifest(copy, sha256=None)

    with pytest.raises(ModelUnavailableError, match="no file hashes"):
        BundleInferencePipeline.from_bundle(copy)


def test_different_classes_are_refused(copy: Path) -> None:
    edit_manifest(copy, classes=["glioma", "other"])

    with pytest.raises(ModelUnavailableError, match="classes"):
        BundleInferencePipeline.from_bundle(copy)


def test_different_feature_names_are_refused(copy: Path) -> None:
    names = json.loads((copy / MANIFEST_FILE).read_text())["radiomic_feature_names"]
    edit_manifest(copy, radiomic_feature_names=[*names[:-1], "renamed"])

    with pytest.raises(ModelUnavailableError, match="feature names"):
        BundleInferencePipeline.from_bundle(copy)


def test_a_manifest_missing_a_section_is_unavailable(copy: Path) -> None:
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    del manifest["preprocessing"]
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))

    with pytest.raises(ModelUnavailableError, match="invalid artifact manifest"):
        BundleInferencePipeline.from_bundle(copy)


def test_serving_follows_the_manifest_and_not_the_defaults(copy: Path) -> None:
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    manifest["preprocessing"]["min_mask_fraction"] = 0.2
    manifest["preprocessing"]["crop_margin"] = 0.1
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))

    preprocessing = load_artifact(copy).preprocessing
    assert preprocessing.min_mask_fraction == 0.2
    assert preprocessing.crop_margin == 0.1


def test_a_skull_stripping_setting_is_refused(copy: Path) -> None:
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    manifest["preprocessing"]["skull_stripping"] = False
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))

    with pytest.raises(
        ModelUnavailableError, match=r"unexpected: \['skull_stripping'\]"
    ):
        load_artifact(copy)


def test_a_missing_mask_fraction_is_refused(copy: Path) -> None:
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    del manifest["preprocessing"]["min_mask_fraction"]
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))

    with pytest.raises(
        ModelUnavailableError, match=r"missing: \['min_mask_fraction'\]"
    ):
        load_artifact(copy)


def test_a_missing_crop_margin_is_refused(copy: Path) -> None:
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    del manifest["preprocessing"]["crop_margin"]
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))

    with pytest.raises(ModelUnavailableError, match=r"missing: \['crop_margin'\]"):
        load_artifact(copy)


def test_a_negative_crop_margin_is_refused(copy: Path) -> None:
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    manifest["preprocessing"]["crop_margin"] = -0.5
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))

    with pytest.raises(ModelUnavailableError, match="crop_margin"):
        load_artifact(copy)


def test_normalization_that_the_backbone_was_not_exported_with_is_refused(
    copy: Path,
) -> None:
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    manifest["preprocessing"]["cnn_mean"] = [0.5, 0.5, 0.5]
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))

    with pytest.raises(ModelUnavailableError, match="normalization"):
        BundleInferencePipeline.from_bundle(copy)


@pytest.mark.parametrize("version", [None, 1])
def test_thresholds_for_other_quality_metrics_are_refused(
    copy: Path, version: int | None
) -> None:
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    thresholds = manifest["preprocessing"]["thresholds"]
    if version is None:
        del thresholds["metrics_version"]
    else:
        thresholds["metrics_version"] = version
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))

    with pytest.raises(ModelUnavailableError, match="quality metrics version"):
        BundleInferencePipeline.from_bundle(copy)


def test_a_missing_quality_threshold_is_refused(copy: Path) -> None:
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    del manifest["preprocessing"]["thresholds"]["min_entropy"]
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))

    with pytest.raises(ModelUnavailableError, match=r"missing: \['min_entropy'\]"):
        load_artifact(copy)


def test_a_missing_radiomics_setting_is_refused(copy: Path) -> None:
    manifest = json.loads((copy / MANIFEST_FILE).read_text())
    del manifest["radiomics"]["lbp_points"]
    (copy / MANIFEST_FILE).write_text(json.dumps(manifest))

    with pytest.raises(ModelUnavailableError, match=r"missing: \['lbp_points'\]"):
        load_artifact(copy)
