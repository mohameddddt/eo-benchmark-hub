"""Kaggle entry point for the scene-classification benchmarks.

Output layout (/kaggle/working/out/<benchmark>/):
  ground_truth/split_manifest_v2.csv   full manifest incl. hidden labels (server install)
  public/split_manifest_v2.csv         ids + splits only (safe to publish)
  models/<model_id>/predictions.csv    test_hidden, with per-class probabilities
  models/<model_id>/robustness.csv     EuroSAT-C variants (EuroSAT only)
  models/<model_id>/card.json          model card + training curves + efficiency
  assets/...                           thumbnails for the website
  summary.json
"""
from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eobench_eval import splits
from eobench_train import classification as C
from eobench_train.common import OUT, SMOKE, elapsed_h, environment, log, pip_install, seed_everything, write_json

KERNEL_URL = "https://www.kaggle.com/code/mohameddddt/eobench-{bench}"


def _zoo(bench: str) -> list[C.Spec]:
    S = C.Spec
    if bench == "eurosat":
        zoo = [
            S("simplecnn-scratch", "SimpleCNN (from scratch)", "simplecnn", "none", "CNN", 64,
              epochs=30, lr=2e-3, batch_size=256, weight_decay=5e-4),
            S("resnet18-in1k-64px", "ResNet-18 @64px (v1 baseline recipe)", "resnet18.tv_in1k",
              "ImageNet-1k", "CNN", 64, epochs=12, lr=1e-3),
            S("mobilenetv3-large", "MobileNetV3-Large", "mobilenetv3_large_100.ra_in1k",
              "ImageNet-1k", "Efficient CNN", 224, epochs=10, lr=1e-3),
            S("efficientnet-b0", "EfficientNet-B0", "efficientnet_b0.ra_in1k",
              "ImageNet-1k", "Efficient CNN", 224, epochs=10, lr=1e-3),
            S("resnet18-in1k", "ResNet-18", "resnet18.tv_in1k", "ImageNet-1k", "CNN", 224,
              epochs=10, lr=5e-4),
            S("resnet18-ssl4eo", "ResNet-18 (SSL4EO-S12 MoCo)", "torchgeo:resnet18",
              "SSL4EO-S12 (MoCo v2, Sentinel-2 RGB)", "CNN", 224, epochs=10, lr=5e-4, norm="probe"),
            S("resnet50-in1k", "ResNet-50", "resnet50.tv2_in1k", "ImageNet-1k", "CNN", 224,
              epochs=10, lr=5e-4),
            S("resnet50-ssl4eo", "ResNet-50 (SSL4EO-S12 MoCo)", "torchgeo:resnet50",
              "SSL4EO-S12 (MoCo v2, Sentinel-2 RGB)", "CNN", 224, epochs=10, lr=5e-4, norm="probe"),
            S("vit-small", "ViT-S/16", "vit_small_patch16_224.augreg_in21k_ft_in1k",
              "ImageNet-21k → 1k", "ViT", 224, epochs=10, lr=1e-4),
            S("convnext-tiny", "ConvNeXt-T", "convnext_tiny.fb_in22k_ft_in1k",
              "ImageNet-22k → 1k", "ConvNeXt", 224, epochs=10, lr=1.5e-4),
        ]
    else:  # resisc45
        zoo = [
            S("mobilenetv3-large", "MobileNetV3-Large", "mobilenetv3_large_100.ra_in1k",
              "ImageNet-1k", "Efficient CNN", 224, epochs=12, lr=1e-3),
            S("efficientnet-b0", "EfficientNet-B0", "efficientnet_b0.ra_in1k",
              "ImageNet-1k", "Efficient CNN", 224, epochs=12, lr=1e-3),
            S("resnet18-in1k", "ResNet-18", "resnet18.tv_in1k", "ImageNet-1k", "CNN", 224,
              epochs=12, lr=5e-4),
            S("resnet50-in1k", "ResNet-50", "resnet50.tv2_in1k", "ImageNet-1k", "CNN", 224,
              epochs=12, lr=5e-4),
            S("resnet50-ssl4eo", "ResNet-50 (SSL4EO-S12 MoCo)", "torchgeo:resnet50",
              "SSL4EO-S12 (MoCo v2, Sentinel-2 RGB)", "CNN", 224, epochs=12, lr=5e-4, norm="probe"),
            S("vit-small", "ViT-S/16", "vit_small_patch16_224.augreg_in21k_ft_in1k",
              "ImageNet-21k → 1k", "ViT", 224, epochs=12, lr=1e-4),
            S("convnext-tiny", "ConvNeXt-T", "convnext_tiny.fb_in22k_ft_in1k",
              "ImageNet-22k → 1k", "ConvNeXt", 224, epochs=12, lr=1.5e-4),
        ]
    if SMOKE:
        keep = {"simplecnn-scratch", "resnet18-in1k-64px", "resnet18-ssl4eo"}
        zoo = [s for s in zoo if s.id in keep] or zoo[:2]
        for s in zoo:
            s.epochs, s.img_size, s.batch_size = 1, 32, 32
    return zoo


def _write_manifests(out: Path, rows: list[dict]) -> None:
    for sub, cols in (("ground_truth", ["sample_id", "relpath", "label", "split"]),
                      ("public", ["sample_id", "split"])):
        p = out / sub / "split_manifest_v2.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def _save_thumb(arr_chw: torch.Tensor | np.ndarray, path: Path, size: int) -> None:
    a = arr_chw.permute(1, 2, 0).numpy() if isinstance(arr_chw, torch.Tensor) else arr_chw
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).resize((size, size), Image.BICUBIC).save(path, quality=90)


def main(bench: str, dataset: str, img_store: int | None, thumb: int) -> None:
    seed_everything(42)
    env = environment()
    log("environment:", env)
    out = OUT / bench
    root = Path(dataset)

    # ---- split (identical code path to the server) ----
    if bench == "eurosat":
        img_root = root if (root / "AnnualCrop").is_dir() else next(
            p for p in root.rglob("*") if p.is_dir() and (p / "AnnualCrop").is_dir())
        rows = splits.eurosat_manifest(img_root)
        classes, dropped = splits.EUROSAT_CLASSES, 0
    else:
        img_root = root
        rows, dropped = splits.resisc45_manifest(root)
        classes = splits.RESISC45_CLASSES
    fp = splits.fingerprint(rows)
    log(f"{bench}: {len(rows)} samples, {dropped} duplicates dropped, fingerprint {fp}")
    _write_manifests(out, rows)

    by_split = {s: [r for r in rows if r["split"] == s] for s in splits.SPLITS}
    if SMOKE:
        by_split = {s: v[: (300 if s == "train" else 80)] for s, v in by_split.items()}
    cidx = {c: i for i, c in enumerate(classes)}
    t = time.time()
    X = {s: C.load_images([img_root / r["relpath"] for r in v], img_store) for s, v in by_split.items()}
    Y = {s: np.array([cidx[r["label"]] for r in v]) for s, v in by_split.items()}
    ids = {s: [r["sample_id"] for r in v] for s, v in by_split.items()}
    log(f"decoded images in {time.time() - t:.0f}s:", {s: tuple(x.shape) for s, x in X.items()})

    # ---- EuroSAT-C: corrupt the hidden split once, reuse for every model ----
    robust_x, robust_ids = None, None
    if bench == "eurosat":
        from eobench_eval.corruptions import all_variants, corrupt, variant_id

        t = time.time()
        hid = X["test_hidden"].permute(0, 2, 3, 1).numpy()
        chunks, robust_ids = [], []
        for c, s in all_variants():
            chunks.append(np.stack([corrupt(hid[i], ids["test_hidden"][i], c, s) for i in range(len(hid))]))
            robust_ids += [variant_id(sid, c, s) for sid in ids["test_hidden"]]
        robust_x = torch.from_numpy(np.concatenate(chunks)).permute(0, 3, 1, 2).contiguous()
        log(f"EuroSAT-C: {len(robust_ids)} corrupted variants in {time.time() - t:.0f}s")

    # ---- the zoo ----
    if any(s.arch.startswith("torchgeo:") for s in _zoo(bench)) and not SMOKE:
        pip_install("torchgeo")
    summary = {"benchmark": bench, "fingerprint": fp, "n_samples": len(rows), "duplicates_dropped": dropped,
               "environment": env, "kernel_url": KERNEL_URL.format(bench=bench), "models": []}
    public_probs: dict[str, np.ndarray] = {}
    for spec in _zoo(bench):
        if elapsed_h() > 10.5:
            log("time budget reached — skipping", spec.id)
            continue
        log(f"=== {spec.id} ({spec.arch}, {spec.img_size}px, {spec.epochs} ep)")
        try:
            seed_everything(42)
            model, info = C.train(spec, X["train"], Y["train"], X["val"], Y["val"], len(classes))
            mdir = out / "models" / spec.id
            p_hidden = C.predict_probs(model, X["test_hidden"], spec.img_size, spec.norm)
            p_public = C.predict_probs(model, X["test_public"], spec.img_size, spec.norm)
            public_probs[spec.id] = p_public
            C.write_predictions(mdir / "predictions.csv", ids["test_hidden"], p_hidden, classes)
            if robust_x is not None:
                p_rob = C.predict_probs(model, robust_x, spec.img_size, spec.norm)
                clean = np.concatenate([p_hidden, p_rob])
                C.write_predictions(mdir / "robustness.csv", ids["test_hidden"] + robust_ids, clean,
                                    classes, with_probs=False)
            card = C.model_card(spec, model, info, env,
                                test_public_acc=float((p_public.argmax(1) == Y["test_public"]).mean()),
                                kernel_url=summary["kernel_url"])
            write_json(mdir / "card.json", card)
            summary["models"].append({k: card[k] for k in ("id", "name", "val_acc", "test_public_acc",
                                                           "num_params", "gflops", "throughput_img_s",
                                                           "train_time_s")})
            log(f"  -> val {card['val_acc']:.4f} public {card['test_public_acc']:.4f} "
                f"params {card['num_params'] / 1e6:.1f}M thr {card['throughput_img_s']:.0f} img/s")
            del model
            torch.cuda.empty_cache()
        except Exception:
            log(f"!!! {spec.id} failed:\n{traceback.format_exc()}")
            summary.setdefault("failed", []).append(spec.id)
        write_json(out / "summary.json", summary)

    # ---- website assets ----
    log("writing assets")
    assets = out / "assets"
    rng = np.random.default_rng(0)
    exemplars = {}
    for c in classes:
        idx = np.flatnonzero(Y["train"] == cidx[c])
        pick = rng.choice(idx, size=min(6, len(idx)), replace=False)
        exemplars[c] = []
        for j, i in enumerate(pick):
            name = f"classes/{c}_{j}.jpg"
            _save_thumb(X["train"][i], assets / name, thumb)
            exemplars[c].append(name)

    gallery = []
    if public_probs:
        best = max(summary["models"], key=lambda m: m["val_acc"])["id"]
        pb = public_probs[best]
        pred, conf = pb.argmax(1), pb.max(1)
        wrong = np.flatnonzero(pred != Y["test_public"])
        hard = wrong[np.argsort(-conf[wrong])][:24]                     # confident mistakes
        right = []
        for c in classes:                                               # one confident hit per class
            ok = np.flatnonzero((pred == Y["test_public"]) & (Y["test_public"] == cidx[c]))
            if len(ok):
                right.append(ok[np.argmax(conf[ok])])
        for i in list(hard) + right:
            sid = ids["test_public"][i]
            _save_thumb(X["test_public"][i], assets / "gallery" / f"{sid}.jpg", thumb)
            gallery.append({
                "sample_id": sid, "image": f"gallery/{sid}.jpg", "label": classes[Y["test_public"][i]],
                "kind": "mistake" if i in set(hard) else "correct",
                "predictions": {m: {"label": classes[int(p[i].argmax())], "confidence": float(p[i].max())}
                                for m, p in public_probs.items()},
            })
    corruption_examples = []
    if bench == "eurosat":
        from eobench_eval.corruptions import CORRUPTIONS, SEVERITIES, corrupt

        for k, i in enumerate([3, 11]):  # two fixed public tiles
            base = X["test_public"][i].permute(1, 2, 0).numpy()
            sid = ids["test_public"][i]
            _save_thumb(base, assets / "corruptions" / f"{k}_clean.jpg", thumb)
            for c in CORRUPTIONS:
                for s in SEVERITIES:
                    _save_thumb(corrupt(base, sid, c, s), assets / "corruptions" / f"{k}_{c}_{s}.jpg", thumb)
            corruption_examples.append({"index": k, "label": classes[Y["test_public"][i]]})
    write_json(assets / "assets.json", {"classes": exemplars, "gallery": gallery,
                                        "corruption_examples": corruption_examples, "thumb_px": thumb})
    write_json(out / "summary.json", summary)
    log("done.", json.dumps(summary["models"], indent=1, default=float)[:3000])
