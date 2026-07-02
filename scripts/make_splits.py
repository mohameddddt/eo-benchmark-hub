#!/usr/bin/env python3
"""Deterministic benchmark splits — MUST stay byte-identical to the logic in
the baseline notebooks (Section 4). Do not modify without bumping split_version.

Usage:
  python scripts/make_splits.py --data-dir /path/to/EuroSAT_RGB \
      --out data/ground_truth/eurosat/split_manifest_v1.csv
"""
import argparse
import csv
import hashlib
import random
from pathlib import Path

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
FRACTIONS = {"train": 0.70, "val": 0.10, "test_public": 0.10, "test_hidden": 0.10}
SEED = 42


def make_split_manifest(img_root: Path, classes, fractions: dict, seed: int):
    """Identical logic to the notebook — do not modify."""
    rows = []
    order = ["train", "val", "test_public", "test_hidden"]
    for cls in classes:
        files = sorted((img_root / cls).glob("*.jpg"))
        rng = random.Random(f"{seed}-{cls}")
        idx = list(range(len(files)))
        rng.shuffle(idx)
        n = len(files)
        bounds, acc = [], 0.0
        for s in order[:-1]:
            acc += fractions[s]
            bounds.append(round(acc * n))
        splits = ([order[0]] * bounds[0]
                  + [order[1]] * (bounds[1] - bounds[0])
                  + [order[2]] * (bounds[2] - bounds[1])
                  + [order[3]] * (n - bounds[2]))
        for i, split in zip(idx, splits):
            f = files[i]
            rows.append({"sample_id": f.stem, "relpath": f"{cls}/{f.name}",
                         "label": cls, "split": split})
    rows.sort(key=lambda r: r["sample_id"])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    manifest = make_split_manifest(args.data_dir, CLASSES, FRACTIONS, SEED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "relpath", "label", "split"])
        w.writeheader()
        w.writerows(manifest)

    fp = hashlib.sha256(
        "\n".join(f"{r['sample_id']},{r['split']}" for r in manifest).encode()
    ).hexdigest()[:16]
    print(f"Wrote {len(manifest)} rows -> {args.out}")
    print(f"Split fingerprint: {fp}  (display this on the benchmark page)")
