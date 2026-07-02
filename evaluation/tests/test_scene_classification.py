"""Synthetic tests always run; the golden test runs when the real files are present.

To enable the golden test, copy into evaluation/tests/golden/:
  - split_manifest_v1.csv                    (your real EuroSAT manifest)
  - predictions_resnet18-baseline_v1.csv     (the baseline's predictions)
These contain hidden-split labels — golden/ is gitignored on purpose.
"""
import csv
from pathlib import Path

import pytest

from eobench_eval import get_evaluator

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
CONFIG = {"classes": CLASSES, "split_version": "v1"}
GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture()
def evaluator():
    return get_evaluator("scene_classification")


@pytest.fixture()
def tiny(tmp_path):
    """3-sample benchmark: manifest + a perfect and a broken prediction file."""
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    with open(gt_dir / "split_manifest_v1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "relpath", "label", "split"])
        w.writerow(["Forest_1", "Forest/Forest_1.jpg", "Forest", "test_hidden"])
        w.writerow(["River_1", "River/River_1.jpg", "River", "test_hidden"])
        w.writerow(["Highway_1", "Highway/Highway_1.jpg", "Highway", "test_hidden"])
        w.writerow(["Forest_2", "Forest/Forest_2.jpg", "Forest", "train"])

    perfect = tmp_path / "perfect.csv"
    perfect.write_text(
        "sample_id,predicted_label\nForest_1,Forest\nRiver_1,River\nHighway_1,Highway\n"
    )
    one_wrong = tmp_path / "one_wrong.csv"
    one_wrong.write_text(
        "sample_id,predicted_label\nForest_1,Forest\nRiver_1,Highway\nHighway_1,Highway\n"
    )
    broken = tmp_path / "broken.csv"
    broken.write_text(
        "sample_id,predicted_label\nForest_1,Lava\nForest_1,Forest\nGhost_9,Forest\n"
    )
    return gt_dir, perfect, one_wrong, broken


def test_ground_truth_only_hidden_split(evaluator, tiny):
    gt = evaluator.load_ground_truth(tiny[0], CONFIG)
    assert set(gt) == {"Forest_1", "River_1", "Highway_1"}  # train row excluded


def test_validate_accepts_good_file(evaluator, tiny):
    gt = evaluator.load_ground_truth(tiny[0], CONFIG)
    assert evaluator.validate(tiny[1], gt, CONFIG) == []


def test_validate_catches_everything(evaluator, tiny):
    gt = evaluator.load_ground_truth(tiny[0], CONFIG)
    errors = " | ".join(evaluator.validate(tiny[3], gt, CONFIG))
    assert "duplicate" in errors
    assert "missing" in errors
    assert "unknown" in errors
    assert "Invalid label" in errors


def test_evaluate_perfect_and_imperfect(evaluator, tiny):
    gt = evaluator.load_ground_truth(tiny[0], CONFIG)
    assert evaluator.evaluate(tiny[1], gt, CONFIG)["metrics"]["accuracy"] == 1.0
    result = evaluator.evaluate(tiny[2], gt, CONFIG)
    assert result["metrics"]["accuracy"] == pytest.approx(2 / 3)
    cm = result["extra"]["confusion_matrix"]
    river, highway = cm["labels"].index("River"), cm["labels"].index("Highway")
    assert cm["matrix"][river][highway] == 1  # the one River->Highway mistake


@pytest.mark.skipif(
    not (GOLDEN / "split_manifest_v1.csv").exists(), reason="golden files not present"
)
def test_golden_resnet18_baseline(evaluator):
    """The frozen contract: this exact file must always score exactly this."""
    gt = evaluator.load_ground_truth(GOLDEN, CONFIG)
    pred = GOLDEN / "predictions_resnet18-baseline_v1.csv"
    assert evaluator.validate(pred, gt, CONFIG) == []
    result = evaluator.evaluate(pred, gt, CONFIG)
    assert result["metrics"]["accuracy"] == pytest.approx(0.975925925925926, abs=1e-12)
    assert result["metrics"]["f1_macro"] == pytest.approx(0.9747833696421807, abs=1e-12)
    assert result["extra"]["num_samples"] == 2700
