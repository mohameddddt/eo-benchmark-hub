# Contributing

Thanks for helping improve EO-Bench Hub.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r backend/requirements-dev.txt -e evaluation
cd frontend && npm install
```

## Before opening a pull request

```bash
make lint                         # ruff
make test                         # 47+ tests, no services needed
cd frontend && npm run typecheck && npm run build
```

## Ground rules

- **Never commit hidden labels.** `data/` is gitignored; ground truth comes from the Kaggle kernels via `scripts/import_kaggle.py`.
- **Metric changes bump the evaluator version** (`version = "…/0.x.y"` in the evaluator) and need a test with hand-computed expected values.
- **Split changes create a new `split_version`.** `eobench_eval/splits.py` must stay deterministic — the split fingerprint shown on each benchmark page is the contract.
- New benchmarks: add an evaluator (or reuse one), a training entry in `training/`, and a spec in `scripts/seed.py`.
