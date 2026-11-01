"""Deterministic benchmark splits — the single source of truth.

The same functions run in the Kaggle training kernels (which ship this file in
the code dataset), in scripts/make_splits.py and in the test suite, so the
train/val/test membership a researcher reproduces is exactly the one the
server scores against.

split_v2 sample ids
-------------------
v1 used file stems as sample ids (``AnnualCrop_1``), which made the hidden
split worthless: the label is in the id, and "predict the id prefix" scored
100% accuracy without looking at a single pixel. v2 ids are content hashes of
the image bytes (``es-3f9a0c1d2e4b``): reproducible from the official
download, stable across mirrors, and free of label information.

Hidden is not secret
--------------------
All benchmarks are built from public datasets, so a determined cheater can
always recover test labels by matching pixels. The hub's guarantees are
therefore protocol integrity (everyone is scored on the same ids with the same
versioned metric code), no trivial leaks (above), and transparency (model
cards with notebook links; maintainer-reproduced entries carry a Verified
badge).
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from pathlib import Path

SPLITS = ("train", "val", "test_public", "test_hidden")
DEFAULT_FRACTIONS = {"train": 0.70, "val": 0.10, "test_public": 0.10, "test_hidden": 0.10}
SEED = 42

EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                   "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

RESISC45_CLASSES = [
    "airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge",
    "chaparral", "church", "circular_farmland", "cloud", "commercial_area",
    "dense_residential", "desert", "forest", "freeway", "golf_course", "ground_track_field",
    "harbor", "industrial_area", "intersection", "island", "lake", "meadow",
    "medium_residential", "mobile_home_park", "mountain", "overpass", "palace",
    "parking_lot", "railway", "railway_station", "rectangular_farmland", "river",
    "roundabout", "runway", "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
    "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland",
]

LANDCOVERAI_CLASSES = ["background", "building", "woodland", "water", "road"]

# BigEarthNet 19-class nomenclature, alphabetical: prob_00..prob_18 follow this order.
BIGEARTHNET_CLASSES = [
    "Agro-forestry areas", "Arable land", "Beaches, dunes, sands", "Broad-leaved forest",
    "Coastal wetlands", "Complex cultivation patterns", "Coniferous forest",
    "Industrial or commercial units", "Inland waters", "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters", "Mixed forest", "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas", "Pastures", "Permanent crops",
    "Transitional woodland, shrub", "Urban fabric",
]


def content_id(data: bytes, prefix: str) -> str:
    """Label-free, mirror-stable sample id derived from the raw file bytes."""
    return f"{prefix}-{hashlib.sha256(data).hexdigest()[:12]}"


def stratified_assign(n: int, rng_key: str, fractions: dict = DEFAULT_FRACTIONS) -> list[str]:
    """Split labels for n items of one class (items in canonical sorted order).

    Byte-identical to the v1 algorithm (scripts/make_splits.py at split_v1), so
    EuroSAT v2 keeps exactly the v1 partition — only the ids changed.
    """
    rng = random.Random(rng_key)
    idx = list(range(n))
    rng.shuffle(idx)
    bounds, acc = [], 0.0
    for s in SPLITS[:-1]:
        acc += fractions[s]
        bounds.append(round(acc * n))
    order = ([SPLITS[0]] * bounds[0]
             + [SPLITS[1]] * (bounds[1] - bounds[0])
             + [SPLITS[2]] * (bounds[2] - bounds[1])
             + [SPLITS[3]] * (n - bounds[2]))
    out = [""] * n
    for i, split in zip(idx, order):
        out[i] = split
    return out


def _classification_manifest(files_by_class: dict[str, list[Path]], root: Path, prefix: str,
                             seed: int, fractions: dict) -> tuple[list[dict], int]:
    """Shared builder: dedupe by content, assign stratified splits, hash ids.

    Exact byte duplicates are dropped (all copies, if their labels disagree) —
    otherwise one image could sit in train and test_hidden at the same time.
    """
    first_seen: dict[str, tuple[str, Path]] = {}
    conflicted: set[str] = set()
    hashed: dict[str, list[tuple[Path, str]]] = {}
    for cls in sorted(files_by_class):
        for f in sorted(files_by_class[cls], key=lambda p: p.name):
            sid = content_id(f.read_bytes(), prefix)
            if sid in first_seen:
                if first_seen[sid][0] != cls:
                    conflicted.add(sid)
                continue
            first_seen[sid] = (cls, f)
            hashed.setdefault(cls, []).append((f, sid))

    rows, dropped = [], 0
    for cls in sorted(hashed):
        items = [(f, sid) for f, sid in hashed[cls] if sid not in conflicted]
        splits = stratified_assign(len(items), f"{seed}-{cls}", fractions)
        for (f, sid), split in zip(items, splits):
            rows.append({"sample_id": sid, "relpath": f.relative_to(root).as_posix(),
                         "label": cls, "split": split})
    total = sum(len(v) for v in files_by_class.values())
    dropped = total - len(rows)
    rows.sort(key=lambda r: r["sample_id"])
    return rows, dropped


def eurosat_manifest(root: Path, seed: int = SEED,
                     fractions: dict = DEFAULT_FRACTIONS) -> list[dict]:
    """root = folder containing the 10 class folders (EuroSAT RGB, 64x64 JPGs)."""
    files = {c: sorted((root / c).glob("*.jpg")) for c in EUROSAT_CLASSES}
    # v1 ordered files by full path (== by name within a class) — keep that.
    rows, _ = _classification_manifest(files, root, "es", seed, fractions)
    return rows


def resisc45_manifest(root: Path, seed: int = SEED,
                      fractions: dict = DEFAULT_FRACTIONS) -> tuple[list[dict], int]:
    """root = any folder under which <class>/<image>.jpg files live.

    The Kaggle mirror ships an arbitrary train/test pre-split; files are pooled
    per class and re-split so all hub benchmarks follow one protocol.
    """
    files: dict[str, list[Path]] = {c: [] for c in RESISC45_CLASSES}
    for f in root.rglob("*"):
        cls = f.parent.name.lower().replace(" ", "_").replace("-", "_")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"} and cls in files:
            files[cls].append(f)
    empty = [c for c, v in files.items() if not v]
    if empty:
        raise FileNotFoundError(f"RESISC45 classes with no images under {root}: {empty}")
    return _classification_manifest(files, root, "rs", seed, fractions)


def landcoverai_manifest(train: Iterable[str], val: Iterable[str], test: Iterable[str],
                         seed: int = SEED) -> list[dict]:
    """Keeps the official LandCover.ai tile split; official test is halved
    (seeded) into test_public / test_hidden. Tile names carry no label
    information, so they stay as ids."""
    rows = [{"sample_id": t, "split": "train"} for t in sorted(set(train))]
    rows += [{"sample_id": t, "split": "val"} for t in sorted(set(val))]
    test = sorted(set(test))
    rng = random.Random(f"{seed}-landcoverai-test")
    shuffled = test[:]
    rng.shuffle(shuffled)
    public = set(shuffled[: len(shuffled) // 2])
    rows += [{"sample_id": t, "split": "test_public" if t in public else "test_hidden"}
             for t in test]
    rows.sort(key=lambda r: r["sample_id"])
    return rows


def bigearthnet_manifest(patches: Iterable[tuple[str, Iterable[str], str]], fraction: float = 0.10,
                         seed: int = SEED) -> list[dict]:
    """patches: (patch_id, labels, official_split) from reBEN metadata.parquet.

    A seeded `fraction` of each official split is kept; official train/validation
    map to train/val and the official test sample is halved into test_public /
    test_hidden. Reusing the official geographic split keeps reBEN's low spatial
    correlation between training and test patches. Patch names carry no label
    information, so they stay as ids. Labels are '|'-joined (names contain commas).
    """
    by_split: dict[str, list[tuple[str, str]]] = {}
    for pid, labels, split in patches:
        by_split.setdefault(split, []).append((pid, "|".join(sorted(labels))))
    rows = []
    for split, items in sorted(by_split.items()):
        items.sort()
        rng = random.Random(f"{seed}-bigearthnet-{split}")
        keep = rng.sample(items, round(len(items) * fraction))
        keep.sort()
        if split == "test":
            shuffled = keep[:]
            random.Random(f"{seed}-bigearthnet-hidden").shuffle(shuffled)
            public = {pid for pid, _ in shuffled[: len(shuffled) // 2]}
            rows += [{"sample_id": pid, "labels": lab, "split": "test_public" if pid in public else "test_hidden"}
                     for pid, lab in keep]
        else:
            rows += [{"sample_id": pid, "labels": lab, "split": {"train": "train", "validation": "val"}[split]}
                     for pid, lab in keep]
    rows.sort(key=lambda r: r["sample_id"])
    return rows


def fingerprint(rows: Iterable[dict]) -> str:
    """Short hash of (id, split) membership — shown on the benchmark page so
    anyone can confirm they reproduced the same split."""
    body = "\n".join(f"{r['sample_id']},{r['split']}" for r in sorted(rows, key=lambda r: r["sample_id"]))
    return hashlib.sha256(body.encode()).hexdigest()[:16]
