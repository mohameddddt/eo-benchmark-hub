"""LandCover.ai semantic-segmentation zoo trainer + Kaggle entry point.

Tiles are cut exactly like the dataset's official split.py (512×512, row-major
index k per orthophoto, partial edge tiles skipped), so tile ids match
train.txt / val.txt / test.txt.

Output layout (/kaggle/working/out/landcoverai/):
  ground_truth/split_manifest_v2.csv + ground_truth/masks_hidden.zip
  public/split_manifest_v2.csv
  models/<id>/masks.zip   (test_hidden predictions, one <tile>.png per tile)
  models/<id>/card.json
  assets/samples/*.jpg|png, assets/assets.json
"""
from __future__ import annotations

import copy
import csv
import io
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from eobench_eval import splits
from eobench_train.common import OUT, SMOKE, elapsed_h, environment, log, pip_install, seed_everything, write_json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TILE = 512
N_CLASSES = len(splits.LANDCOVERAI_CLASSES)
PALETTE = [(222, 214, 190), (213, 62, 79), (46, 125, 79), (49, 116, 196), (245, 178, 40)]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
KERNEL_URL = "https://www.kaggle.com/code/mohameddddt/eobench-landcoverai"


@dataclass
class SegSpec:
    id: str
    name: str
    arch: str          # smp class name
    encoder: str
    pretraining: str
    family: str
    epochs: int = 10
    lr: float = 3e-4
    batch_size: int = 8
    weight_decay: float = 1e-4
    notes: str = ""
    extra: dict = field(default_factory=dict)


def zoo() -> list[SegSpec]:
    z = [
        SegSpec("unet-resnet34", "U-Net · ResNet-34", "Unet", "resnet34", "ImageNet-1k", "CNN",
                epochs=12, lr=3e-4),
        SegSpec("deeplabv3plus-resnet50", "DeepLabV3+ · ResNet-50", "DeepLabV3Plus", "resnet50",
                "ImageNet-1k", "CNN", epochs=10, lr=3e-4),
        SegSpec("segformer-mitb2", "SegFormer · MiT-B2", "Segformer", "mit_b2", "ImageNet-1k",
                "Transformer", epochs=10, lr=2e-4),
    ]
    if SMOKE:
        z = z[:1]
        for s in z:
            s.epochs, s.batch_size = 1, 2
    return z


# ---------------------------------------------------------------- tiling
def cut_tiles(root: Path, wanted: set[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    import cv2

    img_paths = sorted((root / "images").glob("*.tif"))
    # preallocate: np.stack over ~10k tiles would transiently double ~11 GB of RAM
    imgs = np.empty((len(wanted), TILE, TILE, 3), np.uint8)
    masks = np.empty((len(wanted), TILE, TILE), np.uint8)
    names: list[str] = []
    for ip in img_paths:
        mp = root / "masks" / ip.name
        img = cv2.imread(str(ip))
        mask = cv2.imread(str(mp))
        assert img.shape[:2] == mask.shape[:2], ip
        k = 0
        for y in range(0, img.shape[0], TILE):
            for x in range(0, img.shape[1], TILE):
                it = img[y:y + TILE, x:x + TILE]
                if it.shape[0] == TILE and it.shape[1] == TILE:
                    name = f"{ip.stem}_{k}"
                    if name in wanted:
                        imgs[len(names)] = it[..., ::-1]  # BGR -> RGB
                        masks[len(names)] = mask[y:y + TILE, x:x + TILE, 0]
                        names.append(name)
                k += 1
        log(f"  tiled {ip.name}: {len(names)} tiles so far")
        if SMOKE and len(names) > 60:
            break
    return names, imgs[:len(names)], masks[:len(names)]


# ---------------------------------------------------------------- model / train
def build(spec: SegSpec) -> nn.Module:
    import segmentation_models_pytorch as smp

    cls = getattr(smp, spec.arch)
    return cls(encoder_name=spec.encoder, encoder_weights=None if SMOKE else "imagenet",
               classes=N_CLASSES)


def to_input(x: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 3, 1, 2).float().div_(255)
    return ((x - MEAN.to(x.device)) / STD.to(x.device)).contiguous(memory_format=torch.channels_last)


def augment(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (B,H,W,3) uint8, y: (B,H,W) — per-sample dihedral transform + photometric jitter."""
    xs, ys = [], []
    for i in range(x.shape[0]):
        xi, yi = x[i], y[i]
        k = int(torch.randint(0, 4, (1,)))
        xi, yi = torch.rot90(xi, k, (0, 1)), torch.rot90(yi, k, (0, 1))
        if torch.rand(1) < 0.5:
            xi, yi = xi.flip(1), yi.flip(1)
        xs.append(xi)
        ys.append(yi)
    x = torch.stack(xs).float()
    b = x.shape[0]
    bright = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    x = (x * bright).clamp_(0, 255)
    return x, torch.stack(ys)


def dice_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = logits.float().softmax(1)
    oh = F.one_hot(y, N_CLASSES).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    union = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (union + 1)).mean()


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, bs: int = 8) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = to_input(torch.from_numpy(x[i:i + bs]).to(DEVICE))
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            logits = model(xb)
        out.append(logits.argmax(1).to(torch.uint8).cpu().numpy())
    return np.concatenate(out)


def confusion(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.bincount(gt.astype(np.int64).ravel() * N_CLASSES + pred.ravel(),
                       minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)


def miou(cm: np.ndarray) -> tuple[float, list[float]]:
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou)), [float(v) for v in iou]


def train(spec: SegSpec, imgs: np.ndarray, masks: np.ndarray, tr_idx: np.ndarray,
          val_idx: np.ndarray) -> tuple[nn.Module, dict]:
    model = build(spec).to(DEVICE).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    bs = spec.batch_size
    spe = len(tr_idx) // bs
    total, warm = spec.epochs * spe, min(500, spe)

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, 1 - (step - warm) / max(1, total - warm)) ** 0.9   # poly

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(DEVICE, enabled=DEVICE == "cuda")
    xval, yval = imgs[val_idx], masks[val_idx]
    history, best = [], {"miou": -1.0, "epoch": -1, "state": None}
    t_start = time.time()
    for epoch in range(spec.epochs):
        model.train()
        t0 = time.time()
        perm = tr_idx[torch.randperm(len(tr_idx)).numpy()]
        loss_sum = 0.0
        for i in range(spe):
            idx = np.sort(perm[i * bs:(i + 1) * bs])
            xb, yb = augment(torch.from_numpy(imgs[idx]).to(DEVICE, non_blocking=True),
                             torch.from_numpy(masks[idx]).to(DEVICE).long())
            xb = ((xb.permute(0, 3, 1, 2) / 255 - MEAN.to(DEVICE)) / STD.to(DEVICE)).contiguous(
                memory_format=torch.channels_last)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb) + 0.5 * dice_loss(logits, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item()
        cm = confusion(predict(model, xval), yval)
        m, _ = miou(cm)
        rec = {"epoch": epoch + 1, "train_loss": loss_sum / max(1, spe), "val_miou": m,
               "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0}
        history.append(rec)
        log(f"  {spec.id} ep{epoch + 1}/{spec.epochs} loss {rec['train_loss']:.4f} val mIoU {m:.4f} "
            f"({rec['time_s']:.0f}s)")
        if m > best["miou"]:
            best = {"miou": m, "epoch": epoch + 1,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
    model.load_state_dict(best["state"])
    return model, {"history": history, "best_epoch": best["epoch"], "val_miou": best["miou"],
                   "train_time_s": time.time() - t_start}


@torch.no_grad()
def efficiency(model: nn.Module) -> dict:
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    res = {"num_params": sum(p.numel() for p in model.parameters())}
    try:
        with FlopCounterMode(display=False) as fc:
            model(torch.randn(1, 3, TILE, TILE, device=DEVICE))
        res["gflops"] = fc.get_total_flops() / 1e9
    except Exception as e:
        log("flop count failed", e)
        res["gflops"] = None
    bs, iters = (2, 2) if SMOKE else (8, 20)
    x = torch.randn(bs, 3, TILE, TILE, device=DEVICE).contiguous(memory_format=torch.channels_last)
    with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
        for _ in range(3):
            model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
    res["throughput_img_s"] = bs * iters / (time.perf_counter() - t)
    return res


# ---------------------------------------------------------------- export
def png_bytes(mask: np.ndarray, palette: bool = False) -> bytes:
    im = Image.fromarray(mask.astype(np.uint8), mode="L")
    if palette:
        im = im.convert("P")
        flat = [v for c in PALETTE for v in c]
        im.putpalette(flat + [0] * (768 - len(flat)))
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def write_mask_zip(path: Path, names: list[str], masks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:  # PNG is already compressed
        for n, m in zip(names, masks):
            z.writestr(f"{n}.png", png_bytes(m))


def main(dataset: str) -> None:
    seed_everything(42)
    env = environment()
    log("environment:", env)
    if not SMOKE:
        pip_install("segmentation-models-pytorch")
    root = Path(dataset)
    out = OUT / "landcoverai"
    lists = {s: (root / f"{s}.txt").read_text().split() for s in ("train", "val", "test")}
    rows = splits.landcoverai_manifest(lists["train"], lists["val"], lists["test"])
    fp = splits.fingerprint(rows)
    log(f"landcoverai: {len(rows)} tiles, fingerprint {fp}")
    for sub in ("ground_truth", "public"):
        p = out / sub / "split_manifest_v2.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "split"])
            w.writeheader()
            w.writerows(rows)

    split_of = {r["sample_id"]: r["split"] for r in rows}
    t = time.time()
    names, imgs, masks = cut_tiles(root, set(split_of))
    log(f"cut {len(names)} tiles in {time.time() - t:.0f}s ({imgs.nbytes / 1e9:.1f} GB)")
    if not SMOKE:
        missing = set(split_of) - set(names)
        assert not missing, f"{len(missing)} listed tiles not produced, e.g. {sorted(missing)[:3]}"
    sel = {s: np.array([i for i, n in enumerate(names) if split_of[n] == s]) for s in splits.SPLITS}
    if SMOKE:
        sel = {s: (v[:24] if len(v) else np.array([i for i in range(len(names))][:8])) for s, v in sel.items()}
    # monitor on a fixed 400-tile slice of val each epoch (full val is 1.6k tiles)
    val_idx = sel["val"][:400]
    hid_names = [names[i] for i in sel["test_hidden"]]
    write_mask_zip(out / "ground_truth" / "masks_hidden.zip", hid_names, masks[sel["test_hidden"]])

    summary = {"benchmark": "landcoverai", "fingerprint": fp, "n_tiles": len(rows), "environment": env,
               "kernel_url": KERNEL_URL, "models": []}
    public_preds = {}
    for spec in zoo():
        if elapsed_h() > 10.5:
            log("time budget reached — skipping", spec.id)
            continue
        log(f"=== {spec.id}")
        try:
            seed_everything(42)
            model, info = train(spec, imgs, masks, sel["train"], val_idx)
            pred_pub = predict(model, imgs[sel["test_public"]])
            public_preds[spec.id] = pred_pub
            m_pub, iou_pub = miou(confusion(pred_pub, masks[sel["test_public"]]))
            pred_hid = predict(model, imgs[sel["test_hidden"]])
            mdir = out / "models" / spec.id
            write_mask_zip(mdir / "masks.zip", hid_names, pred_hid)
            card = asdict(spec)
            card.update(efficiency(model))
            card.update(info)
            card.update({
                "img_size": TILE, "test_public_miou": m_pub, "test_public_iou": iou_pub,
                "optimizer": f"AdamW (lr {spec.lr:g}, wd {spec.weight_decay:g}), warm-up + poly(0.9)",
                "loss": "cross-entropy + 0.5 × soft Dice",
                "augmentation": "dihedral (rot90 × flip), brightness ±15%",
                "throughput_setup": f"{env['gpu']}, fp16 autocast, batch 8, 512px",
                "environment": env, "kernel_url": KERNEL_URL,
            })
            write_json(mdir / "card.json", card)
            summary["models"].append({k: card.get(k) for k in ("id", "name", "val_miou", "test_public_miou",
                                                               "num_params", "gflops", "throughput_img_s")})
            log(f"  -> val mIoU {info['val_miou']:.4f} public mIoU {m_pub:.4f}")
            del model
            torch.cuda.empty_cache()
        except Exception:
            log(f"!!! {spec.id} failed:\n{traceback.format_exc()}")
            summary.setdefault("failed", []).append(spec.id)
        write_json(out / "summary.json", summary)

    # ---- assets: diverse public tiles with GT and every model's prediction ----
    assets = out / "assets" / "samples"
    assets.mkdir(parents=True, exist_ok=True)
    pub = sel["test_public"]
    hist = np.stack([np.bincount(masks[i].ravel(), minlength=N_CLASSES) for i in pub]) / TILE ** 2
    ent = -(hist * np.log(hist + 1e-9)).sum(1)
    chosen, seen_src = [], set()
    for j in np.argsort(-ent):
        src = names[pub[j]].rsplit("_", 1)[0]
        if src not in seen_src:
            chosen.append(j)
            seen_src.add(src)
        if len(chosen) == 8:
            break
    samples = []
    for j in chosen:
        n = names[pub[j]]
        Image.fromarray(imgs[pub[j]]).resize((384, 384), Image.BICUBIC).save(assets / f"{n}.jpg", quality=88)
        (assets / f"{n}_gt.png").write_bytes(png_bytes(np.asarray(
            Image.fromarray(masks[pub[j]]).resize((384, 384), Image.NEAREST)), palette=True))
        preds = {}
        for mid, pp in public_preds.items():
            (assets / f"{n}_{mid}.png").write_bytes(png_bytes(np.asarray(
                Image.fromarray(pp[j]).resize((384, 384), Image.NEAREST)), palette=True))
            m, _ = miou(confusion(pp[j], masks[pub[j]]))
            preds[mid] = {"mask": f"samples/{n}_{mid}.png", "miou": m}
        samples.append({"sample_id": n, "image": f"samples/{n}.jpg", "gt": f"samples/{n}_gt.png",
                        "class_fractions": [float(v) for v in hist[j]], "predictions": preds})
    write_json(out / "assets" / "assets.json", {"samples": samples, "palette": PALETTE,
                                                 "classes": splits.LANDCOVERAI_CLASSES})
    write_json(out / "summary.json", summary)
    log("done", summary["models"])
