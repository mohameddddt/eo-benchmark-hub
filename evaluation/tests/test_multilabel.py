"""Synthetic tests for the multi-label evaluator. Golden test gets added
after the first real BigEarthNet baseline submission (same pattern as EuroSAT)."""
import csv

import pytest

from eobench_eval import get_evaluator

CLASSES = ["Arable land", "Coniferous forest", "Inland waters", "Pastures", "Urban fabric"]
CONFIG = {"classes": CLASSES, "split_version": "v1"}
N = len(CLASSES)


def header():
    return ["sample_id"] + [f"prob_{i:02d}" for i in range(N)]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header())
        w.writerows(rows)


@pytest.fixture()
def evaluator():
    return get_evaluator("multilabel_classification")


@pytest.fixture()
def gt_dir(tmp_path):
    d = tmp_path / "gt"
    d.mkdir()
    with open(d / "split_manifest_v1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "relpath", "labels", "split"])
        w.writerow(["p1", "p1.png", "Arable land|Pastures", "test_hidden"])
        w.writerow(["p2", "p2.png", "Coniferous forest", "test_hidden"])
        w.writerow(["p3", "p3.png", "Inland waters|Coniferous forest", "test_hidden"])
        w.writerow(["p4", "p4.png", "Urban fabric", "train"])  # excluded
    return d


def test_ground_truth_only_hidden(evaluator, gt_dir):
    gt = evaluator.load_ground_truth(gt_dir, CONFIG)
    assert set(gt) == {"p1", "p2", "p3"}
    assert gt["p1"] == frozenset({"Arable land", "Pastures"})


def test_perfect_predictions(evaluator, gt_dir, tmp_path):
    gt = evaluator.load_ground_truth(gt_dir, CONFIG)
    p = tmp_path / "perfect.csv"
    write_csv(p, [
        ["p1", 0.9, 0.1, 0.1, 0.9, 0.1],
        ["p2", 0.1, 0.9, 0.1, 0.1, 0.1],
        ["p3", 0.1, 0.9, 0.9, 0.1, 0.1],
    ])
    assert evaluator.validate(p, gt, CONFIG) == []
    res = evaluator.evaluate(p, gt, CONFIG)
    m = res["metrics"]
    assert m["map_macro"] == 1.0 and m["f1_macro"] == 1.0 and m["subset_accuracy"] == 1.0
    # Urban fabric has zero positives in the hidden split -> AP undefined (None)
    assert res["extra"]["per_class"]["Urban fabric"]["ap"] is None


def test_imperfect_predictions(evaluator, gt_dir, tmp_path):
    gt = evaluator.load_ground_truth(gt_dir, CONFIG)
    p = tmp_path / "imperfect.csv"
    write_csv(p, [
        ["p1", 0.9, 0.1, 0.1, 0.2, 0.1],   # misses Pastures at threshold 0.5
        ["p2", 0.1, 0.9, 0.1, 0.1, 0.1],
        ["p3", 0.1, 0.9, 0.9, 0.1, 0.1],
    ])
    res = evaluator.evaluate(p, gt, CONFIG)
    assert res["metrics"]["subset_accuracy"] == pytest.approx(2 / 3)
    assert res["extra"]["per_class"]["Pastures"]["f1"] == 0.0
    # positive still outranks negatives (0.2 > 0.1) -> ranking metrics unaffected
    assert res["metrics"]["map_macro"] == 1.0


def test_ranking_damage_lowers_map(evaluator, gt_dir, tmp_path):
    gt = evaluator.load_ground_truth(gt_dir, CONFIG)
    p = tmp_path / "rankmiss.csv"
    write_csv(p, [
        ["p1", 0.9, 0.1, 0.1, 0.1, 0.1],   # Pastures positive scored 0.1 ...
        ["p2", 0.1, 0.9, 0.1, 0.3, 0.1],   # ... below a negative at 0.3
        ["p3", 0.1, 0.9, 0.9, 0.1, 0.1],
    ])
    assert evaluator.evaluate(p, gt, CONFIG)["metrics"]["map_macro"] < 1.0


def test_validation_catches_structure(evaluator, gt_dir, tmp_path):
    gt = evaluator.load_ground_truth(gt_dir, CONFIG)

    bad_header = tmp_path / "h.csv"
    bad_header.write_text("sample_id,a,b,c,d,e\np1,0,0,0,0,0\n")
    assert any("Bad header" in e for e in evaluator.validate(bad_header, gt, CONFIG))

    out_of_range = tmp_path / "r.csv"
    write_csv(out_of_range, [["p1", 0.5, 0.5, 0.5, 0.5, 1.5],
                             ["p2", 0, 0, 0, 0, 0], ["p3", 0, 0, 0, 0, 0]])
    assert any("outside [0, 1]" in e for e in evaluator.validate(out_of_range, gt, CONFIG))

    incomplete = tmp_path / "i.csv"
    write_csv(incomplete, [["p1", 0.5, 0.5, 0.5, 0.5, 0.5], ["ghost", 0, 0, 0, 0, 0]])
    errs = " | ".join(evaluator.validate(incomplete, gt, CONFIG))
    assert "missing" in errs and "unknown" in errs