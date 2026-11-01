# Assessment of v1 and what v2 changed

This document records an audit of the v1 codebase (commit `564bf6b`) and how
each finding was addressed in v2. Findings are ordered by severity, and each one
comes with evidence that can be re-run.

## Critical

### 1. The hidden test split could be solved without a model
v1 EuroSAT sample ids were file stems such as `AnnualCrop_1`, so every id
contained its label. Predicting the id's prefix scored **100.00 % accuracy on
the 2,700-image "hidden" split**. Reproduce it with:

```python
import csv
rows = [r for r in csv.DictReader(open("data/ground_truth/eurosat/split_manifest_v1.csv"))
        if r["split"] == "test_hidden"]
print(sum(r["sample_id"].rsplit("_", 1)[0] == r["label"] for r in rows) / len(rows))  # 1.0
```

**v2:** ids are content hashes of the image bytes (`es-3f9a0c1d2e4b`). They can
be reproduced from the official download, stay stable across mirrors, and carry
no label information. `split_v2` keeps the exact v1 partition (verified
row-for-row), so only the ids changed. Byte-identical duplicates are removed so
the same image can never appear in both train and test
(`evaluation/eobench_eval/splits.py`, tested in `test_splits_stats_corruptions.py`).

The docs also say plainly that *hidden is not secret*: with public datasets, a
determined cheater can always match pixels. That is why entries reproduced by
the maintainers carry a **Verified** badge.

### 2. It couldn't run without a four-service Docker stack
v1 needed Docker Compose with Postgres, Redis, a Celery worker and a Vite
container just to show a page, and this machine has no Docker. Without a Redis
broker, a successful upload raised an error.

**v2:** defaults to SQLite and an in-process evaluation thread. One `uvicorn`
process serves the API *and* the built React app. Celery and Postgres remain
available through `EVAL_BACKEND=celery` and `docker compose up`. A static export
lets the site be hosted read-only (GitHub Pages workflow included).

### 3. There was almost nothing to present
v1 had one benchmark with data (EuroSAT), one model (ResNet-18), and an empty
leaderboard until someone uploaded a file. BigEarthNet was registered but
inactive: no ground truth, and its notebook needed a 59 GB download.

**v2:** four benchmarks across three task types, each with a model zoo trained on
Kaggle T4 GPUs using one documented recipe per benchmark (`training/`):

| Benchmark | Task | Models |
|---|---|---|
| EuroSAT | scene classification, 10 classes | 10 (scratch CNN → ConvNeXt-T, ImageNet vs SSL4EO-S12 pretraining) |
| EuroSAT-C *(new)* | robustness, 7 corruptions × 3 severities | same 10 |
| NWPU-RESISC45 *(new)* | scene classification, 45 classes | 7 |
| LandCover.ai *(new)* | semantic segmentation, 5 classes | 3 (U-Net, DeepLabV3+, SegFormer) |

Every baseline goes through the same upload → validate → evaluate path as a
user's file. No number on the site is hand-entered.

## Major

| # | v1 weakness | v2 |
|---|---|---|
| 4 | Point estimates only, so a 0.2-point gap on 2,700 images looked like a win | Seeded 95 % bootstrap CIs on every primary metric. The resampling unit fits the task: images, tiles for segmentation, base images for robustness. The leaderboard groups statistically tied entries into **tiers** |
| 5 | Accuracy was the only axis that mattered | Calibration (ECE, NLL, Brier, reliability diagram) when probabilities are uploaded; robustness track; parameters, GFLOPs and measured T4 throughput in every model card, plotted as a Pareto frontier |
| 6 | Best-per-user collapsing would have merged a lab's whole model zoo into one row | One row per (researcher, model name). Resubmitting the *same* model still can't stack the board |
| 7 | Quota counted rejected uploads, which reveal nothing about labels | Only scored submissions count |
| 8 | Frontend: 5 bare pages, `any` throughout, no charts, model cards never shown | Typed React app with a landing page, task-aware analysis (efficiency frontier, CI forest plot, per-class heatmaps, confusion matrices, reliability diagrams, corruption heatmaps, severity curves, mask-comparison slider), a cross-benchmark model matrix, docs, and dark mode |
| 9 | Design doc promised segmentation, admin tools and more that didn't exist | Segmentation evaluator (streaming ZIP-of-PNG, tile-bootstrap mIoU), robustness evaluator, maintainer re-evaluate and verify endpoints. Remaining roadmap items are listed honestly in the README |
| 10 | No domain-pretraining comparison: SSL4EO weights expect reflectance-scaled inputs | Input normalization for domain-pretrained weights is chosen by a frozen-feature kNN probe on train/val (never test) and recorded in the model card |

## Minor

- `passlib` (unmaintained) was replaced with `bcrypt` directly, with SHA-256 pre-hashing so passphrases longer than bcrypt's 72-byte limit still count in full.
- Uploads now have a size cap, and oversized files are deleted rather than left half-written.
- Inline mode re-queues unfinished jobs after a restart.
- Evaluator crashes are no longer retried three times; they are deterministic.
- Tracebacks are only shown to the submission's owner and maintainers.
- The test suite grew from 8 to 44 tests: evaluators, splits, corruptions, bootstrap maths, and the full API flow on SQLite. The frozen v1 golden score (0.975926 / 0.974783) still reproduces exactly under evaluator v0.2.0.

## Known limitations

- Labels for public datasets can't be truly hidden (see 1). Integrity comes from protocol, transparency and verification.
- Baselines use one shared recipe and a single seed each. The CIs cover test-set sampling noise, not training-seed variance.
- RESISC45's licence is research-only. The hub never redistributes its images beyond small thumbnails, and it links to the source.
- BigEarthNet's multi-label evaluator is implemented and tested, but the benchmark isn't seeded: its 59 GB archive doesn't fit the Kaggle workflow.
