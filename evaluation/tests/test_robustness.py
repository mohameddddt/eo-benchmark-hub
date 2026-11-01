import csv

import pytest

from eobench_eval import get_evaluator
from eobench_eval.corruptions import variant_id

CLASSES = ["Forest", "River", "Highway"]
CONFIG = {"classes": CLASSES, "split_version": "v2", "corruptions": ["haze", "clouds"],
          "severities": [1, 2]}
GT = {"es-a": "Forest", "es-b": "River", "es-c": "Highway", "es-d": "Forest"}


@pytest.fixture()
def evaluator():
    return get_evaluator("scene_classification_robustness")


@pytest.fixture()
def gt_dir(tmp_path):
    d = tmp_path / "gt"
    d.mkdir()
    with open(d / "split_manifest_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "relpath", "label", "split"])
        for sid, label in GT.items():
            w.writerow([sid, f"{label}/{sid}.jpg", label, "test_hidden"])
        w.writerow(["es-z", "Forest/z.jpg", "Forest", "train"])
    return d


def write(path, preds):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "predicted_label"])
        w.writerows(preds.items())
    return path


def all_ids(fn):
    out = dict(GT)
    for sid, label in GT.items():
        for c in CONFIG["corruptions"]:
            for s in CONFIG["severities"]:
                out[variant_id(sid, c, s)] = fn(sid, label, c, s)
    return out


def test_perfect_model_is_fully_robust(evaluator, gt_dir, tmp_path):
    gt = evaluator.load_ground_truth(gt_dir, CONFIG)
    f = write(tmp_path / "p.csv", all_ids(lambda sid, label, c, s: label))
    assert evaluator.validate(f, gt, CONFIG) == []
    m = evaluator.evaluate(f, gt, CONFIG)["metrics"]
    assert m["mca"] == 1.0 and m["relative_robustness"] == 1.0


def test_cloud_failure_is_localised(evaluator, gt_dir, tmp_path):
    """Model fails every clouds-2 variant and nothing else."""
    gt = evaluator.load_ground_truth(gt_dir, CONFIG)
    preds = all_ids(lambda sid, label, c, s: "River" if (c, s) == ("clouds", 2) and label != "River"
                    else ("Forest" if (c, s) == ("clouds", 2) else label))
    out = evaluator.evaluate(write(tmp_path / "p.csv", preds), gt, CONFIG)
    per = out["extra"]["per_corruption"]
    assert per["haze"]["mean"] == 1.0
    assert per["clouds"]["s1"] == 1.0 and per["clouds"]["s2"] == 0.0
    assert out["metrics"]["mca"] == pytest.approx(3 / 4)          # 1 of 4 variants fully wrong
    assert out["metrics"]["worst_corruption_accuracy"] == pytest.approx(0.5)
    assert out["metrics"]["clean_accuracy"] == 1.0
    lo, hi = out["ci"]["mca"]
    assert lo <= 0.75 <= hi


def test_missing_variants_rejected(evaluator, gt_dir, tmp_path):
    gt = evaluator.load_ground_truth(gt_dir, CONFIG)
    preds = all_ids(lambda sid, label, c, s: label)
    del preds[variant_id("es-a", "haze", 2)]
    errors = evaluator.validate(write(tmp_path / "p.csv", preds), gt, CONFIG)
    assert any("missing" in e and "es-a@haze-2" in e for e in errors)
