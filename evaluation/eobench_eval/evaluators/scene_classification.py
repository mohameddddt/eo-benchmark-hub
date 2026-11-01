"""Scene-classification evaluator (EuroSAT, RESISC45).

Prediction CSV, one row per test_hidden sample:
    sample_id,predicted_label[,prob_<class_0>,...,prob_<class_K-1>]
Probability columns are optional; when present (all classes, canonical order)
they unlock calibration metrics (ECE, NLL, Brier) and top-2 accuracy.

Frozen contract: the v1 ResNet-18 golden file must keep scoring accuracy
0.975926 / macro F1 0.974783 (tests/test_scene_classification.py).
"""
import csv
import math
from pathlib import Path

import numpy as np

from eobench_eval.base import Evaluator, read_manifest_rows
from eobench_eval.registry import register
from eobench_eval.stats import accuracy_ci, macro_f1_ci

N_CAL_BINS = 15
MAX_ERRORS = 10


class SceneClassificationEvaluator(Evaluator):
    task_type = "scene_classification"
    version = "scene_classification/0.2.0"
    formats = ("csv",)
    metrics = {
        "accuracy": {"label": "Accuracy", "higher_is_better": True, "unit": "ratio",
                     "description": "Fraction of hidden-split images whose predicted label is correct."},
        "f1_macro": {"label": "Macro F1", "higher_is_better": True, "unit": "ratio",
                     "description": "Unweighted mean of per-class F1 — every class counts equally."},
        "balanced_accuracy": {"label": "Balanced acc.", "higher_is_better": True, "unit": "ratio",
                              "description": "Mean per-class recall; robust to class imbalance."},
        "cohen_kappa": {"label": "Cohen's κ", "higher_is_better": True, "unit": "value",
                        "description": "Agreement with ground truth corrected for chance agreement."},
        "top2_accuracy": {"label": "Top-2 acc.", "higher_is_better": True, "unit": "ratio",
                          "description": "True class among the two most probable (needs probabilities)."},
        "ece": {"label": "ECE", "higher_is_better": False, "unit": "ratio",
                "description": f"Expected calibration error, {N_CAL_BINS} equal-width confidence bins: "
                               "how far confidence is from accuracy (needs probabilities)."},
        "nll": {"label": "NLL", "higher_is_better": False, "unit": "value",
                "description": "Mean negative log-likelihood of the true class (needs probabilities)."},
        "brier": {"label": "Brier", "higher_is_better": False, "unit": "value",
                  "description": "Mean squared error between the probability vector and the one-hot "
                                 "truth (needs probabilities)."},
    }

    # ---- spec ----
    def format_spec(self, config: dict) -> dict:
        classes = config["classes"]
        return {
            "file_type": "csv",
            "header": "sample_id,predicted_label[,prob_<class>…]",
            "rows": "one per test_hidden sample id",
            "columns": [
                {"name": "sample_id", "description": "content-hash id from the public split manifest"},
                {"name": "predicted_label", "description": f"one of the {len(classes)} class names"},
                {"name": "prob_<class>", "description": "optional — one column per class in canonical "
                                                        "order, floats in [0, 1] summing to 1; enables "
                                                        "calibration metrics"},
            ],
            "example": "sample_id,predicted_label\n"
                       f"{config.get('id_prefix', 'xx')}-3f9a0c1d2e4b,{classes[0]}\n"
                       f"{config.get('id_prefix', 'xx')}-8b21e07c55aa,{classes[-1]}",
        }

    # ---- ground truth ----
    def load_ground_truth(self, gt_dir: Path, config: dict) -> dict[str, str]:
        """Reads split_manifest_<version>.csv, keeps the test_hidden rows."""
        return {r["sample_id"]: r["label"] for r in read_manifest_rows(gt_dir, config)}

    # ---- shared reader ----
    @staticmethod
    def _read(pred_path: Path, classes: list[str]) -> tuple[list[list[str]], bool, list[str]]:
        base = ["sample_id", "predicted_label"]
        with_probs = base + [f"prob_{c}" for c in classes]
        with open(pred_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header == base:
                return list(reader), False, []
            if header == with_probs:
                return list(reader), True, []
            return [], False, [
                f"Bad header: {header[:4] if header else None}{'…' if header and len(header) > 4 else ''} "
                f"(expected 'sample_id,predicted_label' optionally followed by prob_<class> for all "
                f"{len(classes)} classes in canonical order)"
            ]

    # ---- contract ----
    def validate(self, pred_path: Path, gt: dict[str, str], config: dict) -> list[str]:
        classes = list(config["classes"])
        rows, has_probs, errors = self._read(pred_path, classes)
        if errors:
            return errors
        width = 2 + (len(classes) if has_probs else 0)
        ids = []
        for line, r in enumerate(rows, start=2):
            if len(r) != width:
                errors.append(f"Line {line}: expected {width} columns, got {len(r)}")
            elif has_probs:
                try:
                    p = [float(v) for v in r[2:]]
                except ValueError:
                    errors.append(f"Line {line}: non-numeric probability")
                else:
                    if any(not (0.0 <= v <= 1.0) or math.isnan(v) for v in p):
                        errors.append(f"Line {line}: probability outside [0, 1]")
                    elif abs(sum(p) - 1.0) > 0.02:
                        errors.append(f"Line {line}: probabilities sum to {sum(p):.3f}, expected 1")
            if len(errors) >= MAX_ERRORS:
                return errors + ["(stopping after 10 errors)"]
            if r:
                ids.append(r[0])
        seen = set(ids)
        if len(ids) != len(seen):
            errors.append(f"{len(ids) - len(seen)} duplicate sample_id(s)")
        missing, extra = set(gt) - seen, seen - set(gt)
        if missing:
            errors.append(f"{len(missing)} missing hidden-split id(s), e.g. {sorted(missing)[:3]}")
        if extra:
            errors.append(f"{len(extra)} unknown id(s), e.g. {sorted(extra)[:3]}")
        bad = {r[1] for r in rows if len(r) > 1} - set(classes)
        if bad:
            errors.append(f"Invalid label value(s): {sorted(bad)[:5]}")
        return errors

    def evaluate(self, pred_path: Path, gt: dict[str, str], config: dict) -> dict:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            cohen_kappa_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )

        classes = list(config["classes"])
        cidx = {c: i for i, c in enumerate(classes)}
        rows, has_probs, _ = self._read(pred_path, classes)
        by_id = {r[0]: r for r in rows}
        order = sorted(gt)
        y_true = [gt[s] for s in order]
        y_pred = [by_id[s][1] for s in order]
        yt = np.array([cidx[c] for c in y_true])
        yp = np.array([cidx[c] for c in y_pred])

        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, zero_division=0)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=classes)),
        }
        ci = {
            "accuracy": accuracy_ci((yt == yp).astype(float)),
            "f1_macro": macro_f1_ci(yt, yp, len(classes)),
        }
        extra: dict = {
            "num_samples": len(order),
            "has_probabilities": has_probs,
            "per_class": {
                c: {"precision": float(p), "recall": float(r), "f1": float(s), "support": int(n)}
                for c, p, r, s, n in zip(classes, prec, rec, f1, support)
            },
            "confusion_matrix": {
                "labels": classes,
                "matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
            },
        }
        if has_probs:
            probs = np.array([[float(v) for v in by_id[s][2:]] for s in order])
            probs = probs / probs.sum(axis=1, keepdims=True)
            metrics.update(self._calibration(probs, yt, extra))
        return {"metrics": metrics, "ci": ci, "extra": extra}

    @staticmethod
    def _calibration(probs: np.ndarray, yt: np.ndarray, extra: dict) -> dict:
        n, k = probs.shape
        conf = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == yt).astype(float)
        edges = np.linspace(0, 1, N_CAL_BINS + 1)
        b = np.clip(np.digitize(conf, edges[1:-1]), 0, N_CAL_BINS - 1)
        bins, ece = [], 0.0
        for i in range(N_CAL_BINS):
            m = b == i
            cnt = int(m.sum())
            if cnt:
                acc, cf = float(correct[m].mean()), float(conf[m].mean())
                ece += cnt / n * abs(acc - cf)
                bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "count": cnt,
                             "accuracy": acc, "confidence": cf})
        onehot = np.eye(k)[yt]
        top2 = np.argsort(-probs, axis=1)[:, :2]
        extra["reliability"] = bins
        return {
            "top2_accuracy": float((top2 == yt[:, None]).any(axis=1).mean()),
            "ece": float(ece),
            "nll": float(-np.log(np.clip(probs[np.arange(n), yt], 1e-12, 1)).mean()),
            "brier": float(((probs - onehot) ** 2).sum(axis=1).mean()),
        }


register(SceneClassificationEvaluator())
