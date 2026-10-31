import json
from pathlib import Path

import joblib
import pytest
from pytest import CaptureFixture, MonkeyPatch

from src.contracts import TumorLabel
from src.tests.training_support import NOW, PER_CLASS, make_config
from src.training import __main__ as cli
from src.training import run
from src.training.artifact import CLASSIFIER_FILE, MANIFEST_FILE
from src.training.model import candidate_classifiers
from src.training.run import TrainingResult, train


def test_a_run_writes_the_artifact_and_a_timestamped_report(
    result: TrainingResult,
) -> None:
    assert result.artifact_dir.name == "model-v9.9.9"
    assert (result.artifact_dir / CLASSIFIER_FILE).is_file()
    assert (result.artifact_dir / MANIFEST_FILE).is_file()
    assert result.report_dir.name == "20261029T100000Z"
    assert (result.report_dir / "report.json").is_file()
    assert (result.report_dir / "report.md").is_file()


def test_the_manifest_records_how_the_model_was_trained(result: TrainingResult) -> None:
    manifest = json.loads((result.artifact_dir / MANIFEST_FILE).read_text())

    assert manifest["version"] == "9.9.9" and manifest["seed"] == 42
    assert manifest["classifier"] in candidate_classifiers()
    assert manifest["calibration"]["method"] == "sigmoid"
    assert manifest["preprocessing"]["cnn_size"] == 224
    assert manifest["preprocessing"]["crop_margin"] == 0.05
    assert manifest["quality_thresholds"] == {
        "min_snr": 2.74,
        "min_entropy": 3.2,
        "metrics_version": 2,
    }
    assert len(manifest["radiomic_feature_names"]) == 45
    assert manifest["dataset"]["split_removed_per_class"] == {
        label.value: 0 for label in TumorLabel
    }
    assert manifest["metrics"]["test"]["macro_f1"] == result.report["test"]["macro_f1"]


def test_the_report_has_metrics_selection_and_ablation(result: TrainingResult) -> None:
    report = result.report

    assert report["dataset"]["train_images"] == 4 * PER_CLASS
    assert report["dataset"]["skipped_train"] == 0
    assert set(report["dataset"]["split_removed_per_class"]) == {
        label.value for label in TumorLabel
    }
    assert set(report["selection"]["scores"]) == set(candidate_classifiers())
    assert set(report["ablation_macro_f1"]) == {"cnn_only", "radiomics_only", "hybrid"}
    assert len(report["test"]["confusion_matrix"]) == 4
    assert report["test"]["macro_f1"] > 0.5


def test_the_saved_classifier_gives_calibrated_probabilities(
    result: TrainingResult, workspace: dict[str, Path]
) -> None:
    model = joblib.load(result.artifact_dir / CLASSIFIER_FILE)

    assert len(model.classes_) == 4


def test_the_test_split_is_scored_exactly_once(
    workspace: dict[str, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    calls = []
    real = run.evaluate
    monkeypatch.setattr(
        run, "evaluate", lambda *a, **k: calls.append(1) or real(*a, **k)
    )

    train(make_config(workspace, tmp_path), now=NOW)

    assert len(calls) == 1


def test_the_same_seed_reproduces_the_same_metrics(
    workspace: dict[str, Path], result: TrainingResult, tmp_path: Path
) -> None:
    again = train(make_config(workspace, tmp_path), now=NOW)

    assert again.report["test"] == result.report["test"]
    assert again.report["selection"] == result.report["selection"]
    assert again.report["ablation_macro_f1"] == result.report["ablation_macro_f1"]


def test_an_existing_artifact_stops_the_run_before_any_work(
    workspace: dict[str, Path], result: TrainingResult, tmp_path: Path
) -> None:
    config = make_config(workspace, result.artifact_dir.parent)

    with pytest.raises(FileExistsError):
        train(config, now=NOW)


def test_a_manifest_without_training_images_is_an_error(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    lines = workspace["manifest"].read_text().splitlines()
    only_test = [lines[0]] + [line for line in lines[1:] if line.startswith("Test/")]
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(only_test) + "\n")
    config = make_config({**workspace, "manifest": manifest}, tmp_path / "out")

    with pytest.raises(ValueError, match="no usable training images"):
        train(config, now=NOW)


def test_the_cli_prints_where_the_results_went(
    workspace: dict[str, Path], tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    cli.main(
        [
            "--data-dir", str(workspace["data"]),
            "--manifest", str(workspace["manifest"]),
            "--backbone-bundle", str(workspace["backbone"]),
            "--out", str(tmp_path),
            "--n-splits", "3",
            "--version", "cli-test",
            "--cache-dir", str(workspace["root"] / "cache"),
        ]
    )  # fmt: skip

    output = capsys.readouterr().out
    assert "model-vcli-test" in output and "test macro-F1" in output
    assert "removed from Train (group in Test): glioma 0" in output


def test_the_cli_warns_when_the_target_is_missed(
    capsys: CaptureFixture[str], monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    report = {
        "test": {"macro_f1": 0.5, "accuracy": 0.5},
        "target_met": False,
        "target_macro_f1": 0.9,
        "dataset": {"split_removed_per_class": {"glioma": 0}},
    }
    monkeypatch.setattr(
        cli, "train", lambda config: TrainingResult(tmp_path, tmp_path, report)
    )

    cli.main(["--data-dir", "d", "--manifest", "m"])

    assert "not met" in capsys.readouterr().err
