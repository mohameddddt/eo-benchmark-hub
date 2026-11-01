"""Semantic-segmentation evaluator (LandCover.ai).

Prediction: a ZIP with one single-channel PNG per test_hidden tile, named
``<tile_id>.png`` (folders inside the zip are ignored). Pixel values are class
indices 0..K-1; mode L or P; same size as the tile (512×512).

Ground truth: ``masks_hidden_<split_version>.zip`` (same layout) next to the
split manifest. Evaluation streams tile by tile, accumulating a per-tile
confusion matrix, so memory stays flat no matter how many tiles there are;
per-tile matrices also drive the tile-level bootstrap CI for mIoU.
"""
import io
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from eobench_eval.base import Evaluator, SubmissionError, read_manifest_rows
from eobench_eval.registry import register
from eobench_eval.stats import miou_ci, miou_from_confusion

SAMPLE_CHECKS = 25


def _decode(data: bytes, name: str, shape: tuple[int, int] | None, k: int) -> np.ndarray:
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise SubmissionError(f"{name}: not a readable PNG ({e})") from None
    if im.mode not in ("L", "P"):
        raise SubmissionError(f"{name}: mode {im.mode}, expected single-channel (L or P) class indices")
    arr = np.asarray(im)
    if shape is not None and arr.shape != shape:
        raise SubmissionError(f"{name}: size {arr.shape[::-1]}, expected {shape[::-1]}")
    if arr.max(initial=0) >= k:
        raise SubmissionError(f"{name}: pixel value {int(arr.max())} ≥ number of classes ({k})")
    return arr


def _index(z: zipfile.ZipFile) -> tuple[dict[str, str], list[str]]:
    """tile id -> member name; plus a list of duplicate ids."""
    out, dupes = {}, []
    for n in z.namelist():
        p = PurePosixPath(n)
        if n.endswith("/") or p.name.startswith(".") or "__MACOSX" in p.parts:
            continue
        if p.suffix.lower() != ".png":
            continue
        if p.stem in out:
            dupes.append(p.stem)
        out[p.stem] = n
    return out, dupes


class SegmentationEvaluator(Evaluator):
    task_type = "semantic_segmentation"
    version = "segmentation/0.1.0"
    formats = ("zip",)
    metrics = {
        "miou": {"label": "mIoU", "higher_is_better": True, "unit": "ratio",
                 "description": "Mean intersection-over-union across classes (dataset-level, "
                                "accumulated over all hidden pixels)."},
        "pixel_accuracy": {"label": "Pixel acc.", "higher_is_better": True, "unit": "ratio",
                           "description": "Fraction of correctly labelled pixels — dominated by "
                                          "the majority class."},
        "mean_f1": {"label": "Mean F1 (Dice)", "higher_is_better": True, "unit": "ratio",
                    "description": "Mean per-class F1 / Dice coefficient."},
        "fw_iou": {"label": "FW-IoU", "higher_is_better": True, "unit": "ratio",
                   "description": "Frequency-weighted IoU (classes weighted by pixel share)."},
    }

    def required_files(self, config: dict) -> list[str]:
        v = config.get("split_version", "v2")
        return [f"split_manifest_{v}.csv", f"masks_hidden_{v}.zip"]

    def format_spec(self, config: dict) -> dict:
        classes = config["classes"]
        return {
            "file_type": "zip",
            "header": "<tile_id>.png per test_hidden tile",
            "rows": "one 512×512 single-channel PNG per test_hidden tile",
            "columns": [{"name": f"pixel = {i}", "description": c} for i, c in enumerate(classes)],
            "example": "predictions.zip\n├── M-33-20-D-c-4-2_17.png\n├── M-33-20-D-c-4-2_42.png\n└── …",
        }

    def load_ground_truth(self, gt_dir: Path, config: dict) -> dict:
        ids = [r["sample_id"] for r in read_manifest_rows(gt_dir, config)]
        zpath = gt_dir / f"masks_hidden_{config.get('split_version', 'v2')}.zip"
        if not zpath.exists():
            raise FileNotFoundError(f"Ground-truth masks missing: {zpath}")
        return {"ids": sorted(ids), "zip": zpath}

    def validate(self, pred_path: Path, gt: dict, config: dict) -> list[str]:
        k = len(config["classes"])
        try:
            z = zipfile.ZipFile(pred_path)
        except zipfile.BadZipFile:
            return ["Not a valid ZIP archive"]
        with z:
            members, dupes = _index(z)
            errors = []
            if dupes:
                errors.append(f"{len(dupes)} duplicate tile(s), e.g. {dupes[:3]}")
            missing, extra = set(gt["ids"]) - set(members), set(members) - set(gt["ids"])
            if missing:
                errors.append(f"{len(missing)} missing tile(s), e.g. {sorted(missing)[:3]}")
            if extra:
                errors.append(f"{len(extra)} unknown tile(s), e.g. {sorted(extra)[:3]}")
            if errors:
                return errors
            # spot-check decoding/shape on a deterministic sample; full check happens in evaluate()
            with zipfile.ZipFile(gt["zip"]) as gz:
                gmembers, _ = _index(gz)
                for tid in gt["ids"][:: max(1, len(gt["ids"]) // SAMPLE_CHECKS)]:
                    shape = _decode(gz.read(gmembers[tid]), tid, None, 256).shape
                    try:
                        _decode(z.read(members[tid]), members[tid], shape, k)
                    except SubmissionError as e:
                        errors.append(str(e))
                        if len(errors) >= 5:
                            break
        return errors

    def evaluate(self, pred_path: Path, gt: dict, config: dict) -> dict:
        classes = list(config["classes"])
        k = len(classes)
        per_tile = np.zeros((len(gt["ids"]), k, k), dtype=np.int64)
        with zipfile.ZipFile(pred_path) as z, zipfile.ZipFile(gt["zip"]) as gz:
            pm, _ = _index(z)
            gm, _ = _index(gz)
            for i, tid in enumerate(gt["ids"]):
                g = _decode(gz.read(gm[tid]), tid, None, 256).astype(np.int64)
                p = _decode(z.read(pm[tid]), pm[tid], g.shape, k).astype(np.int64)
                per_tile[i] = np.bincount(g.ravel() * k + p.ravel(), minlength=k * k).reshape(k, k)
        cm = per_tile.sum(axis=0)
        inter = np.diag(cm).astype(float)
        gt_px, pred_px = cm.sum(1), cm.sum(0)
        union = gt_px + pred_px - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        f1 = np.where(gt_px + pred_px > 0, 2 * inter / np.maximum(gt_px + pred_px, 1), np.nan)
        freq = gt_px / gt_px.sum()
        return {
            "metrics": {
                "miou": float(miou_from_confusion(cm)),
                "pixel_accuracy": float(inter.sum() / cm.sum()),
                "mean_f1": float(np.nanmean(f1)),
                "fw_iou": float(np.nansum(freq * np.nan_to_num(iou))),
            },
            "ci": {"miou": miou_ci(per_tile)},
            "extra": {
                "num_samples": len(gt["ids"]),
                "num_pixels": int(cm.sum()),
                "per_class": {c: {"iou": None if np.isnan(iou[j]) else float(iou[j]),
                                  "f1": None if np.isnan(f1[j]) else float(f1[j]),
                                  "pixel_share": float(freq[j])} for j, c in enumerate(classes)},
                "confusion_matrix": {"labels": classes, "matrix": cm.tolist(), "unit": "pixels"},
            },
        }


register(SegmentationEvaluator())
