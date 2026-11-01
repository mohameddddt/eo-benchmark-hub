#!/usr/bin/env python3
"""Install Kaggle kernel outputs into the local hub.

    python training/kaggle.py pull eurosat       # downloads to data/kaggle/eurosat/
    python scripts/import_kaggle.py eurosat      # installs it

For each benchmark this installs
  * hidden ground truth  -> data/ground_truth/<slug>/       (server-only, gitignored)
  * baseline predictions -> data/baselines/<slug>/<model>/  (scored by scripts/seed.py)
  * website media        -> frontend/public/media/<slug>/   (thumbnails, public manifest)
EuroSAT additionally feeds the EuroSAT-C robustness benchmark.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MEDIA = ROOT / "frontend" / "public" / "media"


def find_output(bench: str, src: Path | None) -> Path:
    candidates = [src / bench] if src else []
    candidates += [DATA / "kaggle" / bench / "out" / bench, DATA / "kaggle" / bench / bench]
    for c in candidates:
        if (c / "summary.json").exists():
            return c
    sys.exit(f"No kernel output for '{bench}' (looked in {[str(c) for c in candidates]}). "
             f"Run: python training/kaggle.py pull {bench}")


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def install(bench: str, out: Path) -> None:
    summary = json.loads((out / "summary.json").read_text())
    gt = DATA / "ground_truth" / bench
    copy(out / "ground_truth" / "split_manifest_v2.csv", gt / "split_manifest_v2.csv")
    if (out / "ground_truth" / "masks_hidden.zip").exists():
        copy(out / "ground_truth" / "masks_hidden.zip", gt / "masks_hidden_v2.zip")
    copy(out / "public" / "split_manifest_v2.csv", MEDIA / bench / "split_manifest_v2.csv")

    media = MEDIA / bench
    if (out / "assets").exists():
        shutil.copytree(out / "assets", media, dirs_exist_ok=True)

    base = DATA / "baselines" / bench
    if base.exists():
        shutil.rmtree(base)
    n = 0
    for mdir in sorted((out / "models").iterdir()):
        card = mdir / "card.json"
        if not card.exists():
            continue
        dst = base / mdir.name
        copy(card, dst / "card.json")
        for name in ("predictions.csv", "masks.zip"):
            if (mdir / name).exists():
                copy(mdir / name, dst / name)
        if (mdir / "robustness.csv").exists():
            rdst = DATA / "baselines" / "eurosat-c" / mdir.name
            copy(mdir / "robustness.csv", rdst / "predictions.csv")
            copy(card, rdst / "card.json")
        n += 1
    (base / "summary.json").write_text(json.dumps(summary, indent=2))
    # let the website label media by display name rather than kernel model id
    assets_json = media / "assets.json"
    if assets_json.exists():
        a = json.loads(assets_json.read_text())
        a["model_names"] = {m["id"]: m["name"] for m in summary.get("models", [])}
        assets_json.write_text(json.dumps(a, indent=1))
    failed = summary.get("failed", [])
    print(f"[{bench}] installed {n} baselines, ground truth, media"
          + (f" — FAILED in kernel: {failed}" if failed else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmarks", nargs="*", default=["eurosat", "resisc45", "landcoverai"])
    ap.add_argument("--src", type=Path, help="folder containing <bench>/summary.json")
    a = ap.parse_args()
    for b in a.benchmarks:
        install(b, find_output(b, a.src))


if __name__ == "__main__":
    main()
