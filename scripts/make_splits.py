#!/usr/bin/env python3
"""Rebuild a benchmark's split_v2 manifest from a local copy of the dataset.

    python scripts/make_splits.py eurosat  --data-dir /path/to/EuroSAT_RGB --out manifest.csv
    python scripts/make_splits.py resisc45 --data-dir /path/to/NWPU-RESISC45 --out manifest.csv

Uses eobench_eval.splits — the same code the Kaggle kernels and the server use —
and prints the split fingerprint shown on the benchmark page, so you can confirm
you reproduced the exact split.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from eobench_eval import splits  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["eurosat", "resisc45"])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--with-labels", action="store_true", help="include relpath/label columns")
    a = ap.parse_args()
    if a.benchmark == "eurosat":
        rows = splits.eurosat_manifest(a.data_dir)
    else:
        rows, dropped = splits.resisc45_manifest(a.data_dir)
        print(f"dropped {dropped} byte-duplicate images")
    cols = ["sample_id", "relpath", "label", "split"] if a.with_labels else ["sample_id", "split"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {a.out}")
    print(f"split fingerprint: {splits.fingerprint(rows)}")
