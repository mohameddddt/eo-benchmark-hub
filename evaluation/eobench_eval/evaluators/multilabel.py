"""Multi-label classification evaluator (BigEarthNet-S2 RGB 10%, 19 classes).

Prediction format — wide CSV of per-class probabilities:
    header: sample_id,prob_00,prob_01,...,prob_18
    prob_XX maps to config["classes"][XX] (canonical alphabetical order);
    values are floats in [0, 1].

Metrics: mAP (macro/micro) from the probabilities, F1 (macro/micro) and
subset accuracy at threshold 0.5. Primary metric: map_macro.

Ground truth manifest columns: sample_id,relpath,labels,split — where labels
is a '|'-joined set (class names contain commas, so ',' is not usable).
"""
import csv
from pathlib import Path

from eobench_eval.base import Evaluator
from eobench_eval.registry import register

THRESHOLD = 0.5


class MultilabelEvaluator(Evaluator):
    task_type = "multilabel_classification"
    version = "multilabel/0.1.0"
    metrics = {
        "map_macro": {"label": "mAP (macro)", "higher_is_better": True, "unit": "ratio",
                      "description": "Mean per-class average precision over classes present in the hidden split."},
        "map_micro": {"label": "mAP (micro)", "higher_is_better": True, "unit": "ratio",
                      "description": "Average precision over all (sample, class) pairs."},
        "f1_macro": {"label": "Macro F1 @0.5", "higher_is_better": True, "unit": "ratio",
                     "description": "Mean per-class F1 at threshold 0.5."},
        "f1_micro": {"label": "Micro F1 @0.5", "higher_is_better": True, "unit": "ratio",
                     "description": "F1 over all (sample, class) decisions at threshold 0.5."},
        "subset_accuracy": {"label": "Subset acc.", "higher_is_better": True, "unit": "ratio",
                            "description": "Fraction of samples whose full label set is exactly right."},
    }

    # ---- ground truth ----
    def load_ground_truth(self, gt_dir: Path, config: dict) -> dict[str, frozenset]:
        manifest = gt_dir / f"split_manifest_{config.get('split_version', 'v1')}.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"Ground truth manifest missing: {manifest}")
        classes = set(config["classes"])
        gt: dict[str, frozenset] = {}
        with open(manifest, newline="") as f:
            for r in csv.DictReader(f):
                if r["split"] != "test_hidden":
                    continue
                labels = frozenset(x for x in r["labels"].split("|") if x)
                unknown = labels - classes
                if unknown:
                    raise ValueError(f"Ground truth contains unknown labels: {sorted(unknown)}")
                gt[r["sample_id"]] = labels
        return gt

    # ---- shared reader ----
    @staticmethod
    def _expected_header(n_classes: int) -> list[str]:
        return ["sample_id"] + [f"prob_{i:02d}" for i in range(n_classes)]

    def _read_predictions(
        self, pred_path: Path, n_classes: int
    ) -> tuple[dict[str, list[float]], list[str]]:
        expected = self._expected_header(n_classes)
        errors: list[str] = []
        preds: dict[str, list[float]] = {}
        with open(pred_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != expected:
                return {}, [
                    f"Bad header (expected sample_id,prob_00..prob_{n_classes - 1:02d}, "
                    f"got {header[:4] if header else None}...)"
                ]
            for line_no, row in enumerate(reader, start=2):
                if len(row) != n_classes + 1:
                    errors.append(f"Line {line_no}: expected {n_classes + 1} columns, got {len(row)}")
                    if len(errors) >= 10:
                        return {}, errors
                    continue
                sid, values = row[0], row[1:]
                try:
                    probs = [float(v) for v in values]
                except ValueError:
                    errors.append(f"Line {line_no}: non-numeric probability")
                    if len(errors) >= 10:
                        return {}, errors
                    continue
                if any(p < 0.0 or p > 1.0 for p in probs):
                    errors.append(f"Line {line_no}: probability outside [0, 1]")
                    if len(errors) >= 10:
                        return {}, errors
                    continue
                if sid in preds:
                    errors.append(f"Duplicate sample_id: {sid}")
                preds[sid] = probs
        return preds, errors

    # ---- contract ----
    def validate(self, pred_path: Path, gt: dict[str, frozenset], config: dict) -> list[str]:
        preds, errors = self._read_predictions(pred_path, len(config["classes"]))
        if errors:
            return errors
        missing, extra = set(gt) - set(preds), set(preds) - set(gt)
        if missing:
            errors.append(f"{len(missing)} missing hidden-split id(s), e.g. {sorted(missing)[:3]}")
        if extra:
            errors.append(f"{len(extra)} unknown id(s), e.g. {sorted(extra)[:3]}")
        return errors

    def evaluate(self, pred_path: Path, gt: dict[str, frozenset], config: dict) -> dict:
        import numpy as np
        from sklearn.metrics import average_precision_score, f1_score

        classes = list(config["classes"])
        preds, _ = self._read_predictions(pred_path, len(classes))
        order = sorted(gt)
        y_true = np.array([[1 if c in gt[s] else 0 for c in classes] for s in order])
        y_score = np.array([preds[s] for s in order])
        y_pred = (y_score >= THRESHOLD).astype(int)

        # AP is undefined for classes with zero positives in the hidden split —
        # macro-average over defined classes only (documented on the benchmark page).
        positives = y_true.sum(axis=0)
        defined = positives > 0
        ap_per_class = {
            classes[j]: float(average_precision_score(y_true[:, j], y_score[:, j]))
            for j in range(len(classes))
            if defined[j]
        }
        map_macro = float(np.mean(list(ap_per_class.values())))
        map_micro = float(average_precision_score(y_true[:, defined].ravel(), y_score[:, defined].ravel()))

        # Macro averages cover only classes with >=1 positive in the hidden split;
        # zero-support classes would otherwise contribute a meaningless F1/AP of 0.
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        f1_macro = float(np.mean(f1_per_class[defined]))
        return {
            "metrics": {
                "map_macro": map_macro,
                "map_micro": map_micro,
                "f1_macro": f1_macro,
                "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
                "subset_accuracy": float((y_pred == y_true).all(axis=1).mean()),
            },
            "extra": {
                "num_samples": len(order),
                "threshold": THRESHOLD,
                "per_class": {
                    classes[j]: {
                        "ap": ap_per_class.get(classes[j]),
                        "f1": float(f1_per_class[j]),
                        "support": int(positives[j]),
                    }
                    for j in range(len(classes))
                },
            },
        }


register(MultilabelEvaluator())