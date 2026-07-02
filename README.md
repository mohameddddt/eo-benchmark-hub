# EO-Benchmark Hub

Earth Observation model benchmarking platform. Researchers train models
(e.g. on Google Colab), upload prediction files, and get scored on a hidden
test split. Public leaderboard, model cards, async evaluation.

Design document: `eo-benchmark-hub-design.md` (v0.3).

## Quickstart

```bash
cp .env.example .env          # then change JWT_SECRET
make dev                      # builds & starts postgres, redis, api, worker, frontend
```

- Frontend: http://localhost:5173
- API + interactive docs: http://localhost:8000/docs

In a second terminal:

```bash
make seed                     # registers the EuroSAT benchmark
```

Then copy your split manifest (the notebook wrote it to
`Drive/eobench/eurosat/split_manifest_v1.csv`) into:

```
data/ground_truth/eurosat/split_manifest_v1.csv
```

Now open the frontend, create an account, open **EuroSAT**, and upload
`predictions_resnet18-baseline_v1.csv`. Expected score: accuracy **0.9759**.

## Phase 0 exit criteria

- [x] `make dev` boots everything
- [x] Register / login / create API token (UI or `/docs`)
- [x] `make test` passes (auth flow + evaluator unit tests)

Bonus (Phase 1 head start): the EuroSAT submission pipeline
(upload → validate → Celery evaluate → results → leaderboard) is wired end to end.

## Layout

```
backend/     FastAPI app (auth, benchmarks, submissions, leaderboard)
evaluation/  eobench-eval package: evaluator plugins + golden tests
worker/      Celery worker running the evaluators
frontend/    React + Vite SPA
scripts/     make_splits.py (deterministic splits), seed.py
data/        storage/ (uploads) and ground_truth/ (server-only labels) — gitignored
```

## Golden tests

Copy `split_manifest_v1.csv` and `predictions_resnet18-baseline_v1.csv` into
`evaluation/tests/golden/` (gitignored) and `make test` will additionally assert
the frozen baseline score: accuracy 0.975926, macro F1 0.974783. If a refactor
changes those digits, a metric bug was introduced.

## Migrations

Phase 0 uses `create_all` on startup (`AUTO_CREATE_TABLES=true`) for zero-friction
dev. When the schema starts evolving, add Alembic: set the flag to false, run
`alembic init`, point `env.py` at `app.database.Base.metadata`, and autogenerate
revisions.

## Colab

Baseline notebook: `notebooks/eurosat_baseline.ipynb`. Colab can't reach
localhost — download the predictions CSV and upload via the web UI, or expose
the API temporarily with ngrok for direct API submission.
