"""Kaggle entry point for BigEarthNet-S2 (RGB, 10% subset, 19-class multi-label).

The full 63 GB archive is attached as a read-only Kaggle dataset and *streamed*:
only the B04/B03/B02 bands of the selected patches are decoded, nothing else
touches the disk. Labels and the official geographic split come from reBEN's
metadata.parquet (Zenodo).

Output layout matches run_classification (ground_truth/, public/, models/,
assets/, summary.json); predictions are `sample_id,prob_00..prob_18`.
"""
from __future__ import annotations

import csv
import io
import json
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eobench_eval import splits
from eobench_train import classification as C
from eobench_train.common import OUT, SMOKE, elapsed_h, environment, log, pip_install, seed_everything, write_json

META_URL = "https://zenodo.org/records/10891137/files/metadata.parquet?download=1"
KERNEL_URL = "https://www.kaggle.com/code/mohameddddt/eobench-bigearthnet"
# L2A reflectance x10000 -> 8-bit: fixed for the benchmark (changing it changes the data)
RGB_CLIP = 2750
BANDS = {"B04": 0, "B03": 1, "B02": 2}
CLASSES = splits.BIGEARTHNET_CLASSES


def zoo() -> list[C.Spec]:
    S = C.Spec
    z = [
        S("mobilenetv3-large", "MobileNetV3-Large", "mobilenetv3_large_100.ra_in1k", "ImageNet-1k",
          "Efficient CNN", 224, epochs=10, lr=1e-3),
        S("efficientnet-b0", "EfficientNet-B0", "efficientnet_b0.ra_in1k", "ImageNet-1k",
          "Efficient CNN", 224, epochs=10, lr=1e-3),
        S("resnet18-in1k", "ResNet-18", "resnet18.tv_in1k", "ImageNet-1k", "CNN", 224, epochs=10, lr=5e-4),
        S("resnet50-in1k", "ResNet-50", "resnet50.tv2_in1k", "ImageNet-1k", "CNN", 224, epochs=10, lr=5e-4),
        S("resnet50-ssl4eo", "ResNet-50 (SSL4EO-S12 MoCo)", "torchgeo:resnet50",
          "SSL4EO-S12 (MoCo v2, Sentinel-2 RGB)", "CNN", 224, epochs=10, lr=5e-4, norm="probe"),
        S("vit-small", "ViT-S/16", "vit_small_patch16_224.augreg_in21k_ft_in1k", "ImageNet-21k → 1k",
          "ViT", 224, epochs=10, lr=1e-4),
        S("convnext-tiny", "ConvNeXt-T", "convnext_tiny.fb_in22k_ft_in1k", "ImageNet-22k → 1k",
          "ConvNeXt", 224, epochs=10, lr=1.5e-4),
    ]
    if SMOKE:
        z = [s for s in z if s.id == "resnet18-in1k"]
        for s in z:
            s.epochs, s.img_size, s.batch_size = 1, 32, 16
    for s in z:
        s.label_smoothing = 0.0  # BCE: smoothing would distort the per-class probabilities
    return z


def load_metadata(path: Path | None) -> list[tuple[str, list[str], str]]:
    import pandas as pd

    if path is None:
        path = Path("/tmp/metadata.parquet")
        if not path.exists():
            log("downloading reBEN metadata.parquet")
            urllib.request.urlretrieve(META_URL, path)
    df = pd.read_parquet(path)
    log(f"metadata: {len(df)} rows, columns {list(df.columns)}")
    for flag in ("contains_seasonal_snow", "contains_cloud_or_shadow"):
        if flag in df.columns:
            n = int(df[flag].sum())
            df = df[~df[flag].astype(bool)]
            log(f"  dropped {n} patches flagged {flag}")
    known = set(CLASSES)
    out, unknown = [], set()
    for pid, labels, split in zip(df["patch_id"], df["labels"], df["split"]):
        labels = [str(x) for x in labels]
        unknown.update(set(labels) - known)
        out.append((str(pid), [x for x in labels if x in known], str(split)))
    if unknown:
        log(f"  WARNING: labels outside the 19-class nomenclature ignored: {sorted(unknown)[:5]}")
    return [o for o in out if o[1]]


def extract_rgb(archive: Path, wanted: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Stream the .tar.zst once; decode B04/B03/B02 for wanted patches only."""
    import tarfile

    import tifffile
    import zstandard

    n = len(wanted)
    out = np.zeros((n, 120, 120, 3), np.uint8)
    got = np.zeros((n, 3), bool)
    need, seen, t0 = 3 * n, 0, time.time()
    with open(archive, "rb") as fh:
        reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(fh, read_size=1 << 24)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for k, m in enumerate(tar):
                if k % 500_000 == 0 and k:
                    log(f"  scanned {k:,} members, {got.sum():,}/{need:,} bands ({(time.time() - t0) / 60:.1f} min)")
                if not m.isfile() or not m.name.endswith(".tif"):
                    continue
                pid, band = m.name.rsplit("/", 1)[-1][:-4].rsplit("_", 1)
                b = BANDS.get(band)
                i = wanted.get(pid) if b is not None else None
                if i is None:
                    continue
                arr = tifffile.imread(io.BytesIO(tar.extractfile(m).read()))
                if arr.shape != (120, 120):
                    arr = np.asarray(Image.fromarray(arr.astype(np.float32)).resize((120, 120), Image.BILINEAR))
                out[i, :, :, b] = np.clip(arr.astype(np.float32) * (255.0 / RGB_CLIP), 0, 255).astype(np.uint8)
                got[i, b] = True
                seen += 1
                if seen == need:
                    log("  all wanted bands found — stopping the scan early")
                    break
    return out, got.all(1)


def main(dataset: str, metadata: str | None = None) -> None:
    seed_everything(42)
    env = environment()
    log("environment:", env)
    if not SMOKE:
        pip_install("zstandard", "tifffile", "imagecodecs", "torchgeo")
    out = OUT / "bigearthnet"
    archive = next(Path(dataset).rglob("*.tar.zst"))
    meta = load_metadata(Path(metadata) if metadata else None)
    rows = splits.bigearthnet_manifest(meta, fraction=0.10 if not SMOKE else 1.0)

    t = time.time()
    index = {r["sample_id"]: i for i, r in enumerate(rows)}
    imgs, ok = extract_rgb(archive, index)
    log(f"extracted {ok.sum()}/{len(rows)} patches in {(time.time() - t) / 60:.1f} min")
    rows = [r for r, good in zip(rows, ok) if good]
    imgs = imgs[ok]
    fp = splits.fingerprint(rows)
    log(f"bigearthnet: {len(rows)} patches, fingerprint {fp}")
    for sub, cols in (("ground_truth", ["sample_id", "labels", "split"]), ("public", ["sample_id", "split"])):
        p = out / sub / "split_manifest_v2.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    cidx = {c: i for i, c in enumerate(CLASSES)}
    x_all = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
    y_all = np.zeros((len(rows), len(CLASSES)), np.float32)
    for i, r in enumerate(rows):
        for lab in r["labels"].split("|"):
            y_all[i, cidx[lab]] = 1
    sel = {s: np.array([i for i, r in enumerate(rows) if r["split"] == s]) for s in splits.SPLITS}
    X = {s: x_all[v] for s, v in sel.items()}
    Y = {s: y_all[v] for s, v in sel.items()}
    ids = {s: [rows[i]["sample_id"] for i in v] for s, v in sel.items()}
    log("split sizes:", {s: len(v) for s, v in sel.items()})

    summary = {"benchmark": "bigearthnet", "fingerprint": fp, "n_samples": len(rows), "rgb_clip": RGB_CLIP,
               "environment": env, "kernel_url": KERNEL_URL, "models": []}
    for spec in zoo():
        if elapsed_h() > 10.5:
            log("time budget reached — skipping", spec.id)
            continue
        log(f"=== {spec.id}")
        try:
            seed_everything(42)
            model, info = C.train(spec, X["train"], Y["train"], X["val"], Y["val"], len(CLASSES))
            p_hidden = C.predict_probs(model, X["test_hidden"], spec.img_size, spec.norm, multilabel=True)
            p_public = C.predict_probs(model, X["test_public"], spec.img_size, spec.norm, multilabel=True)
            mdir = out / "models" / spec.id
            mdir.mkdir(parents=True, exist_ok=True)
            with open(mdir / "predictions.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sample_id"] + [f"prob_{j:02d}" for j in range(len(CLASSES))])
                for sid, pr in zip(ids["test_hidden"], p_hidden):
                    w.writerow([sid] + [f"{v:.6f}" for v in pr])
            card = C.model_card(spec, model, info, env, test_public_map=C.macro_ap(Y["test_public"], p_public),
                                kernel_url=KERNEL_URL)
            write_json(mdir / "card.json", card)
            summary["models"].append({k: card[k] for k in ("id", "name", "val_map", "test_public_map",
                                                           "num_params", "gflops", "throughput_img_s",
                                                           "train_time_s")})
            log(f"  -> val mAP {card['val_map']:.4f} public mAP {card['test_public_map']:.4f}")
            del model
            torch.cuda.empty_cache()
        except Exception:
            log(f"!!! {spec.id} failed:\n{traceback.format_exc()}")
            summary.setdefault("failed", []).append(spec.id)
        write_json(out / "summary.json", summary)

    # ---- website assets: six training patches per class ----
    assets = out / "assets"
    rng = np.random.default_rng(0)
    exemplars = {}
    for c in CLASSES:
        idx = np.flatnonzero(Y["train"][:, cidx[c]] > 0)
        pick = rng.choice(idx, size=min(6, len(idx)), replace=False) if len(idx) else []
        exemplars[c] = []
        for j, i in enumerate(pick):
            name = f"classes/{cidx[c]:02d}_{j}.jpg"
            (assets / "classes").mkdir(parents=True, exist_ok=True)
            Image.fromarray(X["train"][i].permute(1, 2, 0).numpy()).resize((128, 128), Image.BICUBIC).save(
                assets / name, quality=90)
            exemplars[c].append(name)
    label_freq = {c: float(Y["train"][:, cidx[c]].mean()) for c in CLASSES}
    write_json(assets / "assets.json", {"classes": exemplars, "gallery": [], "corruption_examples": [],
                                        "thumb_px": 128, "label_frequency": label_freq,
                                        "mean_labels_per_patch": float(Y["train"].sum(1).mean())})
    write_json(out / "summary.json", summary)
    log("done.", json.dumps(summary["models"], indent=1, default=float)[:3000])
