"""Splits, bootstrap statistics and corruptions — the pieces every score rests on."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from eobench_eval import corruptions, splits, stats


# ---------------------------------------------------------------- splits
def test_stratified_fractions_and_determinism():
    a = splits.stratified_assign(1000, "42-Forest")
    assert a == splits.stratified_assign(1000, "42-Forest")
    counts = {s: a.count(s) for s in splits.SPLITS}
    assert counts == {"train": 700, "val": 100, "test_public": 100, "test_hidden": 100}
    assert a != splits.stratified_assign(1000, "42-River")


def test_content_ids_carry_no_label(tmp_path):
    root = tmp_path / "eurosat"
    for cls in splits.EUROSAT_CLASSES:
        (root / cls).mkdir(parents=True)
        for i in range(10):
            (root / cls / f"{cls}_{i}.jpg").write_bytes(f"{cls}-{i}".encode())
    rows = splits.eurosat_manifest(root)
    assert len(rows) == 100
    for r in rows:
        assert r["sample_id"].startswith("es-") and len(r["sample_id"]) == 15
        assert not any(c.lower() in r["sample_id"].lower() for c in splits.EUROSAT_CLASSES)
    assert splits.fingerprint(rows) == splits.fingerprint(list(reversed(rows)))


def test_duplicate_images_never_straddle_splits(tmp_path):
    root = tmp_path / "r"
    (root / "airplane").mkdir(parents=True)
    (root / "airport").mkdir()
    for i in range(30):
        (root / "airplane" / f"a{i}.jpg").write_bytes(f"a{i}".encode())
        (root / "airport" / f"b{i}.jpg").write_bytes(f"b{i}".encode())
    (root / "airport" / "dup.jpg").write_bytes(b"a3")          # same bytes, other class
    files = {"airplane": sorted((root / "airplane").iterdir()), "airport": sorted((root / "airport").iterdir())}
    rows, dropped = splits._classification_manifest(files, root, "rs", 42, splits.DEFAULT_FRACTIONS)
    ids = [r["sample_id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert dropped == 2   # conflicting duplicate removed from both classes


def test_landcoverai_halves_official_test():
    test = [f"T_{i}" for i in range(100)]
    rows = splits.landcoverai_manifest(["A_1"], ["B_1"], test)
    by = {}
    for r in rows:
        by.setdefault(r["split"], []).append(r["sample_id"])
    assert len(by["test_public"]) == len(by["test_hidden"]) == 50
    assert rows == splits.landcoverai_manifest(["A_1"], ["B_1"], list(reversed(test)))


# ---------------------------------------------------------------- stats
def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(1)
    yt, yp = rng.integers(0, 5, 500), rng.integers(0, 5, 500)
    cm = np.zeros((5, 5), int)
    np.add.at(cm, (yt, yp), 1)
    assert stats.macro_f1_from_confusion(cm) == pytest.approx(f1_score(yt, yp, average="macro"))


def test_ci_brackets_estimate_and_is_reproducible():
    rng = np.random.default_rng(2)
    correct = (rng.random(2700) < 0.97).astype(float)
    ci = stats.accuracy_ci(correct)
    assert ci == stats.accuracy_ci(correct)
    assert ci[0] < correct.mean() < ci[1]
    assert 0.005 < ci[1] - ci[0] < 0.02       # ±~0.65 pt at n=2700, p=0.97


# ---------------------------------------------------------------- corruptions
@pytest.fixture()
def img():
    rng = np.random.default_rng(0)
    base = np.linspace(40, 200, 64, dtype=np.float32)
    return np.clip(base[None, :, None] + rng.normal(0, 10, (64, 64, 3)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", list(corruptions.CORRUPTIONS))
def test_corruptions_deterministic_and_monotone(img, name):
    outs = [corruptions.corrupt(img, "es-abc", name, s) for s in corruptions.SEVERITIES]
    assert all(o.shape == img.shape and o.dtype == np.uint8 for o in outs)
    assert np.array_equal(outs[0], corruptions.corrupt(img, "es-abc", name, 1))
    dist = [np.abs(o.astype(int) - img).mean() for o in outs]
    assert dist[0] > 0 and dist[0] <= dist[1] <= dist[2] * 1.05


def test_variant_id_roundtrip():
    vid = corruptions.variant_id("es-3f9a0c1d2e4b", "band_misregistration", 3)
    assert corruptions.parse_variant_id(vid) == ("es-3f9a0c1d2e4b", "band_misregistration", 3)


def test_macro_ap_ci_brackets_sklearn_estimate():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(3)
    y = (rng.random((800, 6)) < 0.25).astype(int)
    s = np.clip(y * 0.4 + rng.random((800, 6)) * 0.7, 0, 1)
    point = np.mean([average_precision_score(y[:, j], s[:, j]) for j in range(6)])
    lo, hi = stats.macro_ap_ci(y, s, n_boot=300)
    assert lo < point < hi and hi - lo < 0.1


def test_bigearthnet_manifest_keeps_official_split():
    patches = [(f"P{i:04d}", ["Arable land", "Pastures"] if i % 2 else ["Urban fabric"],
                ["train", "validation", "test"][i % 3]) for i in range(3000)]
    rows = splits.bigearthnet_manifest(patches, fraction=0.1)
    assert rows == splits.bigearthnet_manifest(list(reversed(patches)), fraction=0.1)
    by = {}
    for r in rows:
        by.setdefault(r["split"], 0)
        by[r["split"]] += 1
    assert by == {"train": 100, "val": 100, "test_public": 50, "test_hidden": 50}
    assert all("|" in r["labels"] or r["labels"] == "Urban fabric" for r in rows)
