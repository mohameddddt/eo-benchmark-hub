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
PROB_HEADER = "sample_id,predicted_label," + ",".join(f"prob_{c}" for c in CLASSES)


def prob_row(sid, label, conf):
    """One-hot-ish probability row: `conf` on `label`, the rest spread evenly."""
    rest = (1 - conf) / (len(CLASSES) - 1)
    probs = [conf if c == label else rest for c in CLASSES]
    return ",".join([sid, label] + [f"{p:.6f}" for p in probs])
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
    lo, hi = result["ci"]["accuracy"]
    assert lo < 0.975926 < hi and hi - lo < 0.015


def test_probabilities_unlock_calibration(evaluator, tiny, tmp_path):
    gt = evaluator.load_ground_truth(tiny[0], CONFIG)
    f = tmp_path / "probs.csv"
    f.write_text("\n".join([PROB_HEADER, prob_row("Forest_1", "Forest", 0.9),
                             prob_row("River_1", "Highway", 0.6), prob_row("Highway_1", "Highway", 0.9)]))
    assert evaluator.validate(f, gt, CONFIG) == []
    out = evaluator.evaluate(f, gt, CONFIG)
    m = out["metrics"]
    assert m["accuracy"] == pytest.approx(2 / 3)
    # confidence bins: {0.6: wrong} and {0.9, 0.9: right} -> ECE = 1/3*0.6 + 2/3*0.1
    assert m["ece"] == pytest.approx(1 / 3 * 0.6 + 2 / 3 * 0.1, abs=1e-6)
    assert 0 < m["nll"] and 0 < m["brier"] < 2
    assert out["extra"]["has_probabilities"] is True
    assert sum(b["count"] for b in out["extra"]["reliability"]) == 3


def test_labels_only_has_no_calibration(evaluator, tiny):
    gt = evaluator.load_ground_truth(tiny[0], CONFIG)
    out = evaluator.evaluate(tiny[1], gt, CONFIG)
    assert "ece" not in out["metrics"] and out["extra"]["has_probabilities"] is False
    lo, hi = out["ci"]["accuracy"]
    assert lo <= out["metrics"]["accuracy"] <= hi


def test_bad_probabilities_rejected(evaluator, tiny, tmp_path):
    gt = evaluator.load_ground_truth(tiny[0], CONFIG)
    bad = prob_row("River_1", "River", 0.9).replace("0.900000", "1.900000")
    f = tmp_path / "bad.csv"
    f.write_text("\n".join([PROB_HEADER, prob_row("Forest_1", "Forest", 0.9), bad,
                             prob_row("Highway_1", "Highway", 0.9)]))
    assert any("outside [0, 1]" in e for e in evaluator.validate(f, gt, CONFIG))


def test_wrong_header_is_explained(evaluator, tiny, tmp_path):
    gt = evaluator.load_ground_truth(tiny[0], CONFIG)
    f = tmp_path / "h.csv"
    f.write_text("id,label\nForest_1,Forest\n")
    assert "Bad header" in evaluator.validate(f, gt, CONFIG)[0]
