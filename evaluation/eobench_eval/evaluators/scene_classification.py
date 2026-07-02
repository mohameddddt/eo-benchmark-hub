"""Scene-classification evaluator (EuroSAT).

Validated against the golden submission: resnet18-baseline on split_v1
must score accuracy 0.975926 / macro F1 0.974783 (see tests/golden/).
"""
import csv
from pathlib import Path

from eobench_eval.base import Evaluator
from eobench_eval.registry import register


class SceneClassificationEvaluator(Evaluator):
    task_type = "scene_classification"
    version = "scene_classification/0.1.0"

    # ---- ground truth ----
    def load_ground_truth(self, gt_dir: Path, config: dict) -> dict[str, str]:
        """Reads split_manifest_<version>.csv, keeps the test_hidden rows."""
        manifest = gt_dir / f"split_manifest_{config.get('split_version', 'v1')}.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"Ground truth manifest missing: {manifest}")
        with open(manifest, newline="") as f:
            return {
                r["sample_id"]: r["label"]
                for r in csv.DictReader(f)
                if r["split"] == "test_hidden"
            }

    # ---- shared reader ----
    @staticmethod
    def _read_predictions(pred_path: Path) -> tuple[list[dict], list[str]]:
        with open(pred_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["sample_id", "predicted_label"]:
                return [], [
                    f"Bad header: {reader.fieldnames} "
                    "(expected ['sample_id', 'predicted_label'])"
                ]
            return list(reader), []

    # ---- contract ----
    def validate(self, pred_path: Path, gt: dict[str, str], config: dict) -> list[str]:
        rows, errors = self._read_predictions(pred_path)
        if errors:
            return errors
        classes = set(config["classes"])
        ids = [r["sample_id"] for r in rows]
        seen = set(ids)
        if len(ids) != len(seen):
            errors.append(f"{len(ids) - len(seen)} duplicate sample_id(s)")
        missing, extra = set(gt) - seen, seen - set(gt)
        if missing:
            errors.append(f"{len(missing)} missing hidden-split id(s), e.g. {sorted(missing)[:3]}")
        if extra:
            errors.append(f"{len(extra)} unknown id(s), e.g. {sorted(extra)[:3]}")
        bad = {r["predicted_label"] for r in rows} - classes
        if bad:
            errors.append(f"Invalid label value(s): {sorted(bad)}")
        return errors

    def evaluate(self, pred_path: Path, gt: dict[str, str], config: dict) -> dict:
        from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                     precision_recall_fscore_support)

        classes = list(config["classes"])
        rows, _ = self._read_predictions(pred_path)
        preds = {r["sample_id"]: r["predicted_label"] for r in rows}
        order = sorted(gt)
        y_true = [gt[s] for s in order]
        y_pred = [preds[s] for s in order]

        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, zero_division=0
        )
        return {
            "metrics": {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_macro": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
                "f1_micro": float(f1_score(y_true, y_pred, labels=classes, average="micro", zero_division=0)),
            },
            "extra": {
                "num_samples": len(order),
                "per_class": {
                    c: {"precision": float(p), "recall": float(r), "f1": float(s), "support": int(n)}
                    for c, p, r, s, n in zip(classes, prec, rec, f1, support)
                },
                "confusion_matrix": {
                    "labels": classes,
                    "matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
                },
            },
        }


register(SceneClassificationEvaluator())
