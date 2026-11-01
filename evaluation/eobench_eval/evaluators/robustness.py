"""Robustness-track evaluator (EuroSAT-C).

The hidden split is scored clean *and* under every corruption × severity of
eobench_eval.corruptions, which researchers regenerate locally from the clean
images. Prediction CSV (header sample_id,predicted_label) must contain:
  * every clean id            es-3f9a0c1d2e4b
  * every corrupted variant   es-3f9a0c1d2e4b@haze-2

Primary metric is mean corruption accuracy (mCA, averaged over all
corruption × severity variants). Relative robustness = mCA / clean accuracy
separates "accurate" from "robust": a model can top the clean leaderboard and
still lose a third of its accuracy under haze.
"""
import csv
from pathlib import Path

import numpy as np

from eobench_eval.base import Evaluator, read_manifest_rows
from eobench_eval.corruptions import CORRUPTIONS, SEVERITIES, variant_id
from eobench_eval.registry import register
from eobench_eval.stats import percentile_ci, resample_indices


class RobustnessEvaluator(Evaluator):
    task_type = "scene_classification_robustness"
    version = "robustness/0.1.0"
    formats = ("csv",)
    metrics = {
        "mca": {"label": "mCA", "higher_is_better": True, "unit": "ratio",
                "description": "Mean corruption accuracy over all 7 corruptions × 3 severities."},
        "clean_accuracy": {"label": "Clean acc.", "higher_is_better": True, "unit": "ratio",
                           "description": "Accuracy on the uncorrupted hidden split."},
        "relative_robustness": {"label": "Rel. robustness", "higher_is_better": True, "unit": "ratio",
                                "description": "mCA ÷ clean accuracy — share of clean accuracy retained."},
        "worst_corruption_accuracy": {"label": "Worst-case acc.", "higher_is_better": True, "unit": "ratio",
                                      "description": "Accuracy on the hardest corruption (mean over "
                                                     "its severities)."},
        "severity3_accuracy": {"label": "Sev-3 acc.", "higher_is_better": True, "unit": "ratio",
                               "description": "Mean accuracy at the strongest severity."},
    }

    @staticmethod
    def _variants(config: dict) -> list[tuple[str, int]]:
        corr = config.get("corruptions") or list(CORRUPTIONS)
        sev = config.get("severities") or list(SEVERITIES)
        return [(c, s) for c in corr for s in sev]

    def format_spec(self, config: dict) -> dict:
        v = self._variants(config)
        return {
            "file_type": "csv",
            "header": "sample_id,predicted_label",
            "rows": f"every clean test_hidden id plus its {len(v)} corrupted variants "
                    f"(<id>@<corruption>-<severity>)",
            "columns": [
                {"name": "sample_id", "description": "clean id, or <id>@<corruption>-<severity>"},
                {"name": "predicted_label", "description": "one of the class names"},
            ],
            "example": "sample_id,predicted_label\nes-3f9a0c1d2e4b,Forest\n"
                       "es-3f9a0c1d2e4b@haze-2,Forest\nes-3f9a0c1d2e4b@clouds-3,SeaLake",
            "generator": "from eobench_eval.corruptions import corrupt\n"
                         "img_c = corrupt(img_uint8_hwc, sample_id, 'haze', 2)",
        }

    def load_ground_truth(self, gt_dir: Path, config: dict) -> dict[str, str]:
        return {r["sample_id"]: r["label"] for r in read_manifest_rows(gt_dir, config)}

    def _expected_ids(self, gt: dict[str, str], config: dict) -> set[str]:
        ids = set(gt)
        for c, s in self._variants(config):
            ids.update(variant_id(sid, c, s) for sid in gt)
        return ids

    @staticmethod
    def _read(pred_path: Path) -> tuple[dict[str, str], list[str]]:
        with open(pred_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != ["sample_id", "predicted_label"]:
                return {}, [f"Bad header: {header} (expected ['sample_id', 'predicted_label'])"]
            preds, dupes, errors = {}, 0, []
            for line, r in enumerate(reader, start=2):
                if len(r) != 2:
                    errors.append(f"Line {line}: expected 2 columns, got {len(r)}")
                    if len(errors) >= 10:
                        break
                    continue
                if r[0] in preds:
                    dupes += 1
                preds[r[0]] = r[1]
            if dupes:
                errors.append(f"{dupes} duplicate sample_id(s)")
            return preds, errors

    def validate(self, pred_path: Path, gt: dict[str, str], config: dict) -> list[str]:
        preds, errors = self._read(pred_path)
        if errors:
            return errors
        expected = self._expected_ids(gt, config)
        missing, extra = expected - set(preds), set(preds) - expected
        if missing:
            errors.append(f"{len(missing)} missing id(s), e.g. {sorted(missing)[:3]}")
        if extra:
            errors.append(f"{len(extra)} unknown id(s), e.g. {sorted(extra)[:3]}")
        bad = set(preds.values()) - set(config["classes"])
        if bad:
            errors.append(f"Invalid label value(s): {sorted(bad)[:5]}")
        return errors

    def evaluate(self, pred_path: Path, gt: dict[str, str], config: dict) -> dict:
        preds, _ = self._read(pred_path)
        order = sorted(gt)
        classes = list(config["classes"])
        variants = self._variants(config)
        clean = np.array([preds[s] == gt[s] for s in order], dtype=float)
        # (V, N) correctness matrix: rows = corruption-severity variants
        corr = np.array([[preds[variant_id(s, c, sev)] == gt[s] for s in order] for c, sev in variants],
                        dtype=float)
        per = {}
        for c in dict.fromkeys(c for c, _ in variants):
            rows = [i for i, (cc, _) in enumerate(variants) if cc == c]
            per[c] = {f"s{variants[i][1]}": float(corr[i].mean()) for i in rows}
            per[c]["mean"] = float(corr[rows].mean())
        sev3 = [i for i, (_, s) in enumerate(variants) if s == max(s for _, s in variants)]
        clean_acc, mca = float(clean.mean()), float(corr.mean())

        # bootstrap over base images: a resampled image keeps all its variants, so
        # averaging over variants first is exact (every image has the same count)
        idx = resample_indices(len(order))
        mca_ci = percentile_ci(corr.mean(axis=0)[idx].mean(axis=1))

        labels = np.array([gt[s] for s in order])
        per_class = {cl: {"clean": float(clean[labels == cl].mean()),
                          "corrupted": float(corr[:, labels == cl].mean())}
                     for cl in classes if (labels == cl).any()}
        return {
            "metrics": {
                "mca": mca,
                "clean_accuracy": clean_acc,
                "relative_robustness": mca / clean_acc if clean_acc else 0.0,
                "worst_corruption_accuracy": min(v["mean"] for v in per.values()),
                "severity3_accuracy": float(corr[sev3].mean()),
            },
            "ci": {"mca": mca_ci, "clean_accuracy": percentile_ci(clean[idx].mean(axis=1))},
            "extra": {
                "num_samples": len(order),
                "num_variants": len(variants),
                "per_corruption": per,
                "per_class": per_class,
                "corruption_descriptions": {c: CORRUPTIONS[c][0] for c in per},
            },
        }


register(RobustnessEvaluator())
