"""Bootstrap confidence intervals.

Leaderboards built on point estimates over-rank noise: on a 2,700-image test
set, a 0.2-point accuracy gap is well inside sampling error. Every primary
metric therefore ships with a 95% percentile-bootstrap interval, and the
leaderboard groups entries whose intervals overlap the tier leader's into the
same tier. Resampling is seeded so a re-evaluation reproduces the interval
exactly.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

N_BOOT = 1000
SEED = 20240611


def resample_indices(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> list[float]:
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def bootstrap(stat: Callable[[np.ndarray], float], n: int, n_boot: int = N_BOOT) -> list[float]:
    """Generic: stat(idx) evaluates the metric on a resampled index vector."""
    idx = resample_indices(n, n_boot)
    return percentile_ci(np.array([stat(row) for row in idx]))


def accuracy_ci(correct: np.ndarray) -> list[float]:
    idx = resample_indices(len(correct))
    return percentile_ci(correct[idx].mean(axis=1))


def macro_f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    """cm: (..., K, K) rows=true, cols=pred -> macro F1 over classes with support."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    support = cm.sum(-1)
    predicted = cm.sum(-2)
    denom = support + predicted
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    has = support > 0
    return (f1 * has).sum(-1) / np.maximum(has.sum(-1), 1)


def macro_f1_ci(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> list[float]:
    idx = resample_indices(len(y_true))
    codes = y_true[idx] * k + y_pred[idx]                       # (B, n)
    offs = (np.arange(idx.shape[0]) * k * k)[:, None]
    cms = np.bincount((codes + offs).ravel(), minlength=idx.shape[0] * k * k)
    return percentile_ci(macro_f1_from_confusion(cms.reshape(-1, k, k)))


def miou_from_confusion(cm: np.ndarray) -> np.ndarray:
    """cm: (..., K, K) -> mean IoU over classes present in gt or prediction."""
    inter = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    union = cm.sum(-1) + cm.sum(-2) - inter
    valid = union > 0
    iou = np.where(valid, inter / np.maximum(union, 1), 0.0)
    return (iou * valid).sum(-1) / np.maximum(valid.sum(-1), 1)


def miou_ci(per_tile_cm: np.ndarray) -> list[float]:
    """per_tile_cm: (T, K, K). Tiles are the resampling unit (pixels within a
    tile are strongly correlated, so resampling pixels would understate error)."""
    idx = resample_indices(len(per_tile_cm))
    sums = np.concatenate([per_tile_cm[chunk].sum(axis=1)       # chunked: bounded memory
                           for chunk in np.array_split(idx, 10)])
    return percentile_ci(miou_from_confusion(sums))
