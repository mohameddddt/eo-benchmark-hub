"""EuroSAT-C: deterministic, EO-specific image corruptions.

Each corruption models a failure mode that real optical satellite pipelines
hit in production, at three severities. Everything is a pure function of
(image, sample_id, corruption, severity), so researchers regenerate the exact
test set locally with this file — numpy + Pillow only, no GPU.

    from eobench_eval.corruptions import corrupt
    out = corrupt(img_uint8_hwc, "es-3f9a0c1d2e4b", "haze", 2)

Robustness-track sample ids are ``<sample_id>@<corruption>-<severity>``.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image

SEVERITIES = (1, 2, 3)

# name -> (short description shown on the website, per-severity parameter)
CORRUPTIONS: dict[str, tuple[str, tuple]] = {
    "gaussian_noise": ("Sensor read noise (additive Gaussian, σ in DN)", (8.0, 16.0, 28.0)),
    "gaussian_blur": ("Defocus / atmospheric MTF loss (Gaussian PSF, σ in px)", (0.6, 1.0, 1.6)),
    "haze": ("Thin cloud / aerosol haze (spatially varying veil)", (0.25, 0.40, 0.55)),
    "clouds": ("Opaque cumulus occlusion (fraction of tile covered)", (0.15, 0.30, 0.45)),
    "jpeg": ("Downlink compression artefacts (JPEG quality)", (40, 20, 10)),
    "low_resolution": ("Coarser ground sampling distance (downsample factor)", (2, 3, 4)),
    "band_misregistration": ("Band-to-band misregistration (R/B shift in px)", (1, 2, 3)),
}


def variant_id(sample_id: str, corruption: str, severity: int) -> str:
    return f"{sample_id}@{corruption}-{severity}"


def parse_variant_id(vid: str) -> tuple[str, str, int]:
    sample_id, rest = vid.rsplit("@", 1)
    corruption, severity = rest.rsplit("-", 1)
    return sample_id, corruption, int(severity)


def _rng(sample_id: str, corruption: str, severity: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{sample_id}|{corruption}|{severity}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _smooth_field(rng: np.random.Generator, h: int, w: int, cells: int) -> np.ndarray:
    """Low-frequency noise in [0, 1]: random grid upsampled bicubically."""
    grid = rng.random((cells, cells)).astype(np.float32)
    img = Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    k = _gaussian_kernel(sigma)
    r = len(k) // 2
    pad = np.pad(x, ((r, r), (r, r), (0, 0)), mode="reflect")
    tmp = sum(k[i] * pad[:, i:i + x.shape[1]] for i in range(len(k)))
    return sum(k[i] * tmp[i:i + x.shape[0]] for i in range(len(k)))


def corrupt(img: np.ndarray, sample_id: str, corruption: str, severity: int) -> np.ndarray:
    """img: HxWx3 uint8 -> HxWx3 uint8."""
    if corruption not in CORRUPTIONS:
        raise KeyError(f"Unknown corruption '{corruption}'. Available: {sorted(CORRUPTIONS)}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")
    p = CORRUPTIONS[corruption][1][severity - 1]
    rng = _rng(sample_id, corruption, severity)
    h, w = img.shape[:2]
    x = img.astype(np.float32)

    if corruption == "gaussian_noise":
        x = x + rng.normal(0.0, p, size=x.shape)
    elif corruption == "gaussian_blur":
        x = _blur(x, p)
    elif corruption == "haze":
        veil = np.array([205.0, 210.0, 215.0], dtype=np.float32)  # bluish-white airlight
        alpha = p * (0.6 + 0.8 * _smooth_field(rng, h, w, 4))[..., None]
        x = x * (1 - alpha) + veil * np.clip(alpha, 0, 1)
    elif corruption == "clouds":
        field = 0.65 * _smooth_field(rng, h, w, 5) + 0.35 * _smooth_field(rng, h, w, 11)
        thr = np.quantile(field, 1.0 - p)  # exactly fraction p above threshold
        t = np.clip((field - thr) / 0.14 + 0.5, 0, 1)
        soft = (t * t * (3 - 2 * t))[..., None]  # smoothstep: feathered cloud edges
        cloud = 215.0 + 40.0 * _smooth_field(rng, h, w, 13)[..., None]
        x = x * (1 - soft) + cloud * soft
    elif corruption == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=int(p))
        x = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.float32)
    elif corruption == "low_resolution":
        small = Image.fromarray(img).resize((max(1, w // p), max(1, h // p)), Image.BILINEAR)
        x = np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.float32)
    elif corruption == "band_misregistration":
        s = int(p)
        x[..., 0] = np.roll(x[..., 0], (s, s), axis=(0, 1))
        x[..., 2] = np.roll(x[..., 2], (-s, -s), axis=(0, 1))

    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def all_variants() -> list[tuple[str, int]]:
    return [(c, s) for c in CORRUPTIONS for s in SEVERITIES]
