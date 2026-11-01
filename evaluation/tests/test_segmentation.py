import csv
import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from eobench_eval import get_evaluator

CLASSES = ["background", "building", "woodland"]
CONFIG = {"classes": CLASSES, "split_version": "v2"}


def png(arr, mode="L"):
    buf = io.BytesIO()
    im = Image.fromarray(arr.astype(np.uint8))
    (im.convert("RGB") if mode == "RGB" else im).save(buf, format="PNG")
    return buf.getvalue()


def make_zip(path, masks, mode="L", prefix=""):
    with zipfile.ZipFile(path, "w") as z:
        for name, m in masks.items():
            z.writestr(f"{prefix}{name}.png", png(m, mode))
    return path


@pytest.fixture()
def evaluator():
    return get_evaluator("semantic_segmentation")


@pytest.fixture()
def bench(tmp_path):
    d = tmp_path / "gt"
    d.mkdir()
    gt = {
        "t1": np.array([[0, 0, 1, 1]] * 4),
        "t2": np.array([[2, 2, 2, 2]] * 4),
    }
    with open(d / "split_manifest_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "split"])
        w.writerows([["t1", "test_hidden"], ["t2", "test_hidden"], ["t0", "train"]])
    make_zip(d / "masks_hidden_v2.zip", gt)
    return d, gt


def test_required_files(evaluator):
    assert evaluator.required_files(CONFIG) == ["split_manifest_v2.csv", "masks_hidden_v2.zip"]


def test_perfect_prediction(evaluator, bench, tmp_path):
    d, masks = bench
    gt = evaluator.load_ground_truth(d, CONFIG)
    f = make_zip(tmp_path / "p.zip", masks, prefix="preds/")   # folders inside zip are fine
    assert evaluator.validate(f, gt, CONFIG) == []
    m = evaluator.evaluate(f, gt, CONFIG)["metrics"]
    assert m["miou"] == 1.0 and m["pixel_accuracy"] == 1.0


def test_hand_computed_iou(evaluator, bench, tmp_path):
    d, masks = bench
    gt = evaluator.load_ground_truth(d, CONFIG)
    pred = {"t1": np.array([[0, 1, 1, 1]] * 4), "t2": masks["t2"]}   # 4 background px -> building
    out = evaluator.evaluate(make_zip(tmp_path / "p.zip", pred), gt, CONFIG)
    iou = {c: v["iou"] for c, v in out["extra"]["per_class"].items()}
    assert iou["background"] == pytest.approx(4 / 8)    # inter 4, union 8
    assert iou["building"] == pytest.approx(8 / 12)     # inter 8, union 12
    assert iou["woodland"] == 1.0
    assert out["metrics"]["miou"] == pytest.approx((0.5 + 2 / 3 + 1) / 3)
    assert out["metrics"]["pixel_accuracy"] == pytest.approx(28 / 32)
    lo, hi = out["ci"]["miou"]
    assert lo <= out["metrics"]["miou"] <= hi


def test_rgb_masks_and_missing_tiles_rejected(evaluator, bench, tmp_path):
    d, masks = bench
    gt = evaluator.load_ground_truth(d, CONFIG)
    errs = evaluator.validate(make_zip(tmp_path / "rgb.zip", masks, mode="RGB"), gt, CONFIG)
    assert any("single-channel" in e for e in errs)
    errs = evaluator.validate(make_zip(tmp_path / "m.zip", {"t1": masks["t1"]}), gt, CONFIG)
    assert any("missing" in e for e in errs)
    bad = tmp_path / "x.zip"
    bad.write_bytes(b"not a zip")
    assert evaluator.validate(bad, gt, CONFIG) == ["Not a valid ZIP archive"]


def test_out_of_range_class_rejected(evaluator, bench, tmp_path):
    d, masks = bench
    gt = evaluator.load_ground_truth(d, CONFIG)
    pred = {"t1": masks["t1"], "t2": np.full((4, 4), 7)}
    errs = evaluator.validate(make_zip(tmp_path / "p.zip", pred), gt, CONFIG)
    assert any("number of classes" in e for e in errs)
