"""Scene-classification model zoo trainer.

Design choices that matter for a fair comparison:
  * every model sees the same split, the same augmentation and the same
    schedule shape (AdamW, 1 warm-up epoch, cosine) — only lr/epochs/input
    size are per-architecture, and all of it lands in the model card;
  * images live in RAM as uint8 and are augmented on the GPU (one affine
    grid_sample per batch), so data loading never bottlenecks a T4;
  * the checkpoint kept is the best *validation* epoch — test splits are only
    touched once, at the end.
"""
from __future__ import annotations

import copy
import csv
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from eobench_train.common import SMOKE, log

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# Input normalizations. Domain-pretrained weights (SSL4EO) were trained on
# reflectance-scaled inputs that 8-bit RGB JPEGs do not map onto exactly, so
# for those the normalization is chosen by a frozen-feature kNN probe on
# train/val (never test) — see pick_norm().
NORMS = {
    "imagenet": (MEAN, STD),
    "unit": (torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)),
    "reflectance": (torch.zeros(1, 3, 1, 1), torch.full((1, 3, 1, 1), 1 / 0.275)),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Spec:
    id: str
    name: str
    arch: str               # timm model name | "simplecnn" | "torchgeo:resnet18" ...
    pretraining: str        # "none" | "ImageNet-1k" | "ImageNet-21k → 1k" | "SSL4EO-S12 (MoCo v2)"
    family: str             # "CNN" | "ViT" | "ConvNeXt" | "Efficient CNN"
    img_size: int
    epochs: int = 10
    lr: float = 3e-4
    batch_size: int = 128
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    norm: str = "imagenet"  # key of NORMS, or "probe"
    notes: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- data
def load_images(paths: list[Path], size: int | None, workers: int = 16) -> torch.Tensor:
    """Decode to an (N, 3, H, W) uint8 tensor, optionally resizing to size×size."""
    def one(p):
        im = Image.open(p).convert("RGB")
        if size and im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        return np.asarray(im)

    with ThreadPoolExecutor(workers) as ex:
        arrs = list(ex.map(one, paths, chunksize=64))
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()


def gpu_batch(x: torch.Tensor, out: int, train: bool, norm: str = "imagenet") -> torch.Tensor:
    """uint8 (B,3,H,W) on device -> normalized float, augmented if train."""
    x = x.float().div_(255)
    b = x.shape[0]
    if train:
        dev = x.device
        scale = torch.empty(b, device=dev).uniform_(0.70, 1.0)       # random zoom-in crop
        k = torch.randint(0, 4, (b,), device=dev).float() * (math.pi / 2)  # rot90 (EO is orientation-free)
        fx = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
        tx = (1 - scale) * torch.empty(b, device=dev).uniform_(-1, 1)
        ty = (1 - scale) * torch.empty(b, device=dev).uniform_(-1, 1)
        cos, sin = torch.cos(k), torch.sin(k)
        theta = torch.stack([
            torch.stack([scale * cos * fx, -scale * sin, tx], 1),
            torch.stack([scale * sin * fx, scale * cos, ty], 1),
        ], 1)
        grid = F.affine_grid(theta, (b, 3, out, out), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        bright = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.9, 1.1)
        contrast = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.9, 1.1)
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (((x - m) * contrast + m) * bright).clamp_(0, 1)
    elif x.shape[-1] != out:
        x = F.interpolate(x, size=(out, out), mode="bilinear", align_corners=False,
                          antialias=x.shape[-1] > out)
    mean, std = NORMS[norm]
    x = (x - mean.to(x.device)) / std.to(x.device)
    return x.contiguous(memory_format=torch.channels_last)


# ---------------------------------------------------------------- models
class SimpleCNN(nn.Module):
    """From-scratch lower anchor: 4 conv blocks, ~0.6M params."""

    def __init__(self, n_classes: int):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 192))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2),
                                  nn.Linear(192, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(spec: Spec, n_classes: int) -> nn.Module:
    import timm

    if spec.arch == "simplecnn":
        return SimpleCNN(n_classes)
    if spec.arch.startswith("torchgeo:"):
        import torchgeo.models as tgm

        name = spec.arch.split(":", 1)[1]                       # resnet18 | resnet50
        weights = getattr(tgm, f"ResNet{name.removeprefix('resnet')}_Weights").SENTINEL2_RGB_MOCO
        model = getattr(tgm, name)(weights=weights)            # timm ResNet, 1000-way head
        model.reset_classifier(n_classes)
        spec.extra["weights_url"] = weights.url
        return model
    return timm.create_model(spec.arch, pretrained=not SMOKE, num_classes=n_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def gflops(model: nn.Module, img_size: int) -> float | None:
    from torch.utils.flop_counter import FlopCounterMode

    try:
        model.eval()
        with FlopCounterMode(display=False) as fc:
            model(torch.randn(1, 3, img_size, img_size, device=DEVICE))
        return fc.get_total_flops() / 1e9
    except Exception as e:  # some exotic ops are not countable — never fail a run on it
        log("flop count failed:", e)
        return None


@torch.no_grad()
def throughput(model: nn.Module, img_size: int, batch: int = 128, iters: int = 25) -> float:
    """Inference images/second (fp16 autocast, channels_last) on this GPU."""
    model.eval()
    if SMOKE:
        batch, iters = 4, 2
    x = torch.randn(batch, 3, img_size, img_size, device=DEVICE).contiguous(memory_format=torch.channels_last)
    with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
        for _ in range(5):
            model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
    return batch * iters / (time.perf_counter() - t)


# ---------------------------------------------------------------- train / predict
@torch.no_grad()
def predict_probs(model: nn.Module, x: torch.Tensor, img_size: int, norm: str = "imagenet",
                  bs: int = 256, multilabel: bool = False) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = gpu_batch(x[i:i + bs].to(DEVICE, non_blocking=True), img_size, train=False, norm=norm)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            logits = model(xb)
        out.append((torch.sigmoid(logits.float()) if multilabel else F.softmax(logits.float(), dim=1)).cpu())
    return torch.cat(out).numpy()


def macro_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mean average precision over classes with at least one positive (multi-label)."""
    from sklearn.metrics import average_precision_score

    has = y_true.sum(0) > 0
    return float(np.mean([average_precision_score(y_true[:, j], scores[:, j]) for j in np.flatnonzero(has)]))


@torch.no_grad()
def pick_norm(spec: Spec, x_tr, y_tr, x_val, y_val, n_classes: int) -> tuple[str, dict]:
    """Frozen-backbone 20-NN accuracy (cosine) on 3k train / 1k val per normalization."""
    model = build_model(spec, n_classes).to(DEVICE).eval()
    model.reset_classifier(0)
    g = torch.Generator().manual_seed(0)
    tr = torch.randperm(len(x_tr), generator=g)[:3000]
    va = torch.randperm(len(x_val), generator=g)[:1000]
    ytr, yva = torch.as_tensor(y_tr)[tr].to(DEVICE), torch.as_tensor(y_val)[va].to(DEVICE)

    def feats(x, norm):
        fs = []
        for i in range(0, len(x), 256):
            xb = gpu_batch(x[i:i + 256].to(DEVICE), spec.img_size, train=False, norm=norm)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                fs.append(F.normalize(model(xb).float(), dim=1))
        return torch.cat(fs)

    scores = {}
    for norm in NORMS:
        ftr, fva = feats(x_tr[tr], norm), feats(x_val[va], norm)
        nn_idx = (fva @ ftr.T).topk(20, dim=1).indices
        if ytr.ndim == 2:   # multi-label: neighbours' mean label vector, scored by macro mAP
            scores[norm] = macro_ap(yva.cpu().numpy(), ytr[nn_idx].float().mean(1).cpu().numpy())
        else:
            votes = F.one_hot(ytr[nn_idx], n_classes).sum(1)
            scores[norm] = float((votes.argmax(1) == yva).float().mean())
    best = max(scores, key=scores.get)
    log(f"  {spec.id} normalization probe (20-NN val score): {scores} -> {best}")
    return best, scores


def _param_groups(model: nn.Module, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def train(spec: Spec, x_tr, y_tr, x_val, y_val, n_classes: int) -> tuple[nn.Module, dict]:
    if spec.norm == "probe":
        spec.norm, spec.extra["norm_probe_knn_acc"] = pick_norm(spec, x_tr, y_tr, x_val, y_val, n_classes)
    model = build_model(spec, n_classes).to(DEVICE).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(_param_groups(model, spec.weight_decay), lr=spec.lr)
    bs = spec.batch_size
    steps_per_epoch = math.ceil(len(x_tr) / bs)
    total, warm = spec.epochs * steps_per_epoch, steps_per_epoch

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(DEVICE, enabled=DEVICE == "cuda")
    multilabel = np.ndim(y_tr) == 2
    crit = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss(label_smoothing=spec.label_smoothing)
    key = "val_map" if multilabel else "val_acc"
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32 if multilabel else torch.long)
    history, best = [], {key: -1.0, "epoch": -1, "state": None}
    t_start = time.time()

    for epoch in range(spec.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(x_tr))
        loss_sum = correct = seen = 0
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            xb = gpu_batch(x_tr[idx].to(DEVICE, non_blocking=True), spec.img_size, train=True, norm=spec.norm)
            yb = y_tr_t[idx].to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                logits = model(xb)
                loss = crit(logits, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(idx)
            if not multilabel:
                correct += (logits.argmax(1) == yb).sum().item()
            seen += len(idx)

        probs = predict_probs(model, x_val, spec.img_size, spec.norm, multilabel=multilabel)
        rec = {"epoch": epoch + 1, "train_loss": loss_sum / seen}
        if multilabel:
            p = np.clip(probs, 1e-7, 1 - 1e-7)
            rec["val_loss"] = float(-(y_val * np.log(p) + (1 - y_val) * np.log(1 - p)).mean())
            rec["val_map"] = macro_ap(y_val, probs)
        else:
            rec["train_acc"] = correct / seen
            rec["val_loss"] = float(-np.log(np.clip(probs[np.arange(len(y_val)), y_val], 1e-12, 1)).mean())
            rec["val_acc"] = float((probs.argmax(1) == y_val).mean())
        rec.update({"lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0})
        history.append(rec)
        log(f"  {spec.id} ep{epoch + 1}/{spec.epochs} loss {rec['train_loss']:.4f} "
            f"{key} {rec[key]:.4f} ({rec['time_s']:.0f}s)")
        if rec[key] > best[key]:
            best = {key: rec[key], "epoch": epoch + 1,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}

    model.load_state_dict(best["state"])
    return model, {"history": history, "best_epoch": best["epoch"], key: best[key],
                   "train_time_s": time.time() - t_start}


# ---------------------------------------------------------------- export
def write_predictions(path: Path, ids: list[str], probs: np.ndarray, classes: list[str],
                      with_probs: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = probs.argmax(1)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "predicted_label"] + ([f"prob_{c}" for c in classes] if with_probs else []))
        for i, sid in enumerate(ids):
            row = [sid, classes[pred[i]]]
            if with_probs:
                row += [f"{p:.6f}" for p in probs[i]]
            w.writerow(row)


def model_card(spec: Spec, model: nn.Module, train_info: dict, env: dict, **extra) -> dict:
    card = asdict(spec)
    card.update({
        "num_params": count_params(model),
        "gflops": gflops(model, spec.img_size),
        "throughput_img_s": throughput(model, spec.img_size),
        "throughput_setup": f"{env['gpu']}, fp16 autocast, batch 128, {spec.img_size}px",
        "optimizer": f"AdamW (lr {spec.lr:g}, wd {spec.weight_decay:g}), 1 warm-up epoch + cosine",
        "augmentation": "random zoom-crop (70–100%), rot90, h-flip, brightness/contrast ±10%",
        "environment": env,
        **train_info,
        **extra,
    })
    return card
