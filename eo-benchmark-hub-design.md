# EO-Benchmark Hub — Full System Design

**Earth Observation Model Benchmarking Platform (SaaS)**
Version 0.3 — Design Document (all decisions resolved — final)

> **Decision log:**
> **D1** FastAPI · **D2** Not hosted, local Docker Compose · **D3** EuroSAT ships first · **D4** Custom re-split with server-only hidden labels (§3.4) · **D5** Quota as config, default 20/week · **D6** Segmentation = ZIP of per-tile PNG masks · **D7** Email+password auth · **D8** **Multi-user** — full auth, API tokens, quotas, and per-user leaderboards stay in scope · **D9** Tiny baselines per benchmark (ResNet-18, small U-Net), SpaceNet baseline optional · **D10** No deadline — **full scope**, nothing cut (SpaceNet and the `eobench` CLI stay in v1) · **D11** Monorepo · **D12** Keep all uploads

---

## 1. Vision & Scope

A web platform where researchers upload predictions from their Earth Observation (EO) ML models and get them automatically evaluated against standard benchmarks (EuroSAT, BigEarthNet, LandCover.ai, SpaceNet). A public leaderboard displays results, fostering reproducibility and fair comparison.

**Key design constraint:** model *training and inference happen outside the platform* (you and other researchers use Google Colab). The platform only receives **prediction files** and evaluates them against held-out ground truth. This keeps the platform cheap to run — no GPUs needed server-side.

### In scope (v1)
- Multi-benchmark support with standardized evaluation protocols
- Prediction upload: CSV, JSON, GeoTIFF (task-dependent)
- Automated async evaluation (accuracy, F1, mIoU, mAP, etc.)
- Public leaderboard with filtering (task, dataset, model type)
- Model cards (architecture, params, training details, Colab notebook link)
- User accounts, submission history, rate limits

### Out of scope (v1)
- Running user model code server-side (code submissions)
- GPU inference on the platform
- Dataset hosting/redistribution (link to official sources instead)
- Paid tiers / billing (design for it, don't build it)

---

## 2. Actors

| Actor | Capabilities |
|---|---|
| Visitor (anonymous) | Browse leaderboards, view model cards, read docs |
| Researcher (registered) | Everything above + submit predictions, manage own submissions/model cards |
| Benchmark Maintainer | Add/update benchmarks, ground truth, evaluation configs |
| Admin | User management, moderation, flag/remove submissions, re-run evaluations |

---

## 3. Core Workflows

### 3.1 Researcher workflow (the Colab loop)

```
┌─────────────────────────  GOOGLE COLAB (researcher side)  ─────────────────────────┐
│ 1. pip install eobench-cli (your small helper package)                             │
│ 2. eobench download eurosat --split test-public   → gets test images (NO labels)   │
│ 3. Train model on train/val splits (GPU on Colab)                                  │
│ 4. Run inference on the test split                                                 │
│ 5. eobench export --format csv → predictions.csv in the required schema            │
│ 6. eobench submit predictions.csv --benchmark eurosat --token $API_KEY             │
│    (or manual upload via the website)                                              │
└────────────────────────────────────────────────────────────────────────────────────┘
                                        │  HTTPS upload
                                        ▼
┌─────────────────────────────  EO-BENCHMARK HUB (server)  ──────────────────────────┐
│ 7. API validates file (schema, size, completeness) → stores in object storage      │
│ 8. Submission row created (status: QUEUED) → task pushed to Redis                  │
│ 9. Celery worker pulls task → loads ground truth → computes metrics                 │
│ 10. Results written to PostgreSQL (status: COMPLETED / FAILED with reason)         │
│ 11. Leaderboard updates; researcher notified (email / dashboard)                   │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Submission state machine

```
UPLOADED → VALIDATING → QUEUED → EVALUATING → COMPLETED
              │                       │
              ▼                       ▼
        REJECTED (bad schema)    FAILED (eval error, retryable)
COMPLETED → PUBLISHED (user opts in to leaderboard) or PRIVATE
```

### 3.3 Benchmark maintainer workflow
1. Register a benchmark: name, task type, description, license, official dataset link.
2. Upload ground-truth labels for the hidden test split (stored encrypted, never downloadable).
3. Define the evaluation config (YAML): expected file format/schema, metrics, primary metric, class list.
4. Publish → benchmark appears on the site with a submission-format spec page and a starter Colab notebook.

### 3.4 Your custom re-split protocol (D4 — decided)

Since the official datasets publish all labels, you'll create your own splits. Do this once per benchmark, deterministically:

1. Download the full official dataset.
2. Split with a **fixed seed**, stratified by class (per-tile for segmentation to avoid spatial leakage between neighboring tiles):
   `train (70%) / val (10%) / test-public (10%, labels released for local sanity checks) / test-hidden (10%, labels server-only)`.
3. Write a **split manifest** per benchmark: `sample_id, split, sha256` — publish the manifest *without* hidden labels so anyone can reproduce exactly which images belong to the hidden split.
4. Store hidden labels under `./data/ground_truth/<benchmark>/` — mounted into the worker container only, never the API container, never in the repo (gitignored).
5. Version the manifest (`split_v1`); results always record which split version they were evaluated on.
6. Keep the splitting script in the repo (`scripts/make_splits.py`) so the whole thing is reproducible from the official download.

For BigEarthNet, prefer the official train/val/test split and hold out your hidden set from *within* the official test split, so your numbers stay roughly comparable to published papers.

### 3.5 Anti-cheating protocol (standardized evaluation)
- **Hidden test labels**: only images are public; labels live server-side only.
- **Submission quota**: e.g. 5 submissions / benchmark / user / week (prevents test-set probing).
- **Checksum + row-count validation** before queueing.
- Optionally, a **public dev split** (labels released) for local sanity checks, and a **private test split** for the leaderboard.

---

## 4. Requirements

### 4.1 Functional requirements

**Auth & accounts**
- FR-1: Email+password registration with verification; optional GitHub/Google OAuth.
- FR-2: Personal API tokens for CLI/Colab submissions.
- FR-3: Profile page: affiliation, links, list of published submissions.

**Benchmarks**
- FR-4: Benchmark registry with task types: `scene_classification`, `multilabel_classification`, `semantic_segmentation`, `object_extraction`.
- FR-5: Each benchmark exposes: description, dataset link, submission format spec, metric definitions, example files, starter Colab notebook link.

**Submissions**
- FR-6: Upload via web UI (drag & drop) and REST API (multipart).
- FR-7: Accepted formats per task:
  - Scene classification (EuroSAT): CSV `sample_id,predicted_label` or JSON.
  - Multi-label (BigEarthNet): CSV `sample_id,label_1;label_2;...` or JSON with probabilities.
  - Segmentation (LandCover.ai): ZIP of GeoTIFF/PNG masks, one per tile, or a single multipage format.
  - Building/road extraction (SpaceNet): GeoJSON polygons/linestrings per tile.
- FR-8: Synchronous validation (schema, sample coverage ≥ configurable %, file size caps) with human-readable error messages.
- FR-9: Async evaluation via Celery; user sees live status; failures show diagnostics.
- FR-10: Rate limiting per user per benchmark.

**Metrics (automated, per task type)**
- FR-11: Classification: accuracy, macro/micro F1, per-class F1, confusion matrix.
- FR-12: Multi-label: micro/macro F1, mAP, precision@k, subset accuracy.
- FR-13: Segmentation: mIoU, per-class IoU, pixel accuracy, Dice.
- FR-14: Object extraction: SpaceNet F1 (IoU-thresholded polygon matching), APLS for roads (v2, it's complex).

**Leaderboard**
- FR-15: Public leaderboard per benchmark, sortable by primary metric, filterable by model type, backbone, params, training data (external data yes/no), date.
- FR-16: One "best" published entry per user per benchmark by default (others collapsible).
- FR-17: Each entry links to its model card.

**Model cards**
- FR-18: Structured form: architecture, backbone, #params, input size, pretraining, training data used, augmentations, hardware ("Google Colab T4/A100"), training time, license, paper/repo/notebook links, free-text notes.
- FR-19: Model card required before a submission can be published to the leaderboard.

**Admin**
- FR-20: Django-admin-style panel: manage users, benchmarks, submissions; trigger re-evaluation (e.g. after fixing a metric bug — versioned evaluators).

### 4.2 Non-functional requirements

| Category | Requirement |
|---|---|
| Performance | Leaderboard pages < 500 ms (cached); validation feedback < 10 s; segmentation eval jobs may take minutes — that's fine, it's async |
| Scale (v1 target) | ~1k users, ~100 submissions/day, files up to 2 GB (segmentation ZIPs) |
| Security | Ground-truth labels never exposed via any API; signed URLs for uploads; API tokens hashed; OWASP basics |
| Reliability | Celery retries (3x, exponential backoff); idempotent evaluators; dead-letter handling |
| Reproducibility | Evaluator code versioned; every result stores `evaluator_version`; re-runs possible |
| Observability | Structured logs, Sentry (errors), Flower (Celery), basic Prometheus metrics |
| Portability | Everything in Docker Compose for dev; single-VM or managed deploy for prod |

---

## 5. System Architecture

**Deployment target (locked):** everything runs on your machine via a single `docker-compose up`. This removes NGINX/TLS, CDN, presigned-URL complexity, and managed cloud services from v1. The design keeps the same *shape* as a real SaaS (so it demos well and could be hosted later), but with local substitutions:

| SaaS component | Local substitution |
|---|---|
| S3 object storage | Local filesystem volume (`./data/storage/`) behind a small `Storage` interface — swap to S3/MinIO later without touching business logic |
| NGINX / Traefik + TLS | None — FastAPI on `localhost:8000`, Vite dev server on `localhost:5173` proxying `/api` |
| Presigned direct-to-storage uploads | Plain multipart upload to FastAPI (fine locally; streaming to disk, not RAM) |
| Email verification / OAuth | Simple email+password, no verification; API tokens still exist for the Colab → localhost flow (see note below) |
| CDN for the SPA | `npm run dev`, or `vite build` served by FastAPI's StaticFiles for the "prod-like" compose profile |

**Colab → local machine caveat:** Colab can't reach `localhost`. For submitting from Colab you have three options: (a) download predictions from Colab and upload through the web UI (simplest, recommended), (b) expose your API temporarily with ngrok/cloudflared when you want CLI submission to work end-to-end, or (c) run the `eobench validate` step in Colab (pure local check) and only `submit` from your machine. The starter notebooks should default to (a).

Original SaaS-shaped diagram (still the conceptual model):

```
                        ┌──────────────────────────────┐
                        │   Researcher (Google Colab)  │
                        │   train → infer → export     │
                        └──────┬───────────────┬───────┘
                               │ CLI/API       │ Browser
                               ▼               ▼
                      ┌─────────────────────────────────┐
                      │        NGINX / Traefik          │  TLS, rate limit, large-upload buffering
                      └───────┬─────────────────┬───────┘
                              │                 │
                    ┌─────────▼──────┐  ┌───────▼────────┐
                    │  React SPA     │  │  FastAPI (or   │
                    │  (static, CDN) │  │  Django) API   │
                    └────────────────┘  └───┬───────┬────┘
                                            │       │
                 ┌──────────────────────────┘       └──────────────┐
                 ▼                                                 ▼
        ┌────────────────┐    enqueue     ┌──────────┐    ┌────────────────┐
        │  PostgreSQL    │◄──────────────►│  Redis   │    │ Object storage │
        │  users, subs,  │                │ broker + │    │ (S3/MinIO)     │
        │  results, meta │                │ cache    │    │ uploads + GT   │
        └───────▲────────┘                └────┬─────┘    └───────▲────────┘
                │                              │ pull task        │
                │        ┌─────────────────────▼──────────────────┴───┐
                │        │        Celery workers (Docker)             │
                └────────┤  validate → load GT → compute metrics      │
                 write   │  (scikit-learn, torchmetrics, rasterio,    │
                 results │   numpy, shapely for SpaceNet)             │
                         └────────────────────────────────────────────┘
```

### 5.1 Component responsibilities

**API service (FastAPI recommended — see Decisions):**
- Auth (JWT sessions + long-lived API tokens for Colab CLI)
- CRUD: benchmarks, submissions, model cards, users
- Presigned upload URLs (client uploads big files straight to object storage; API never buffers 2 GB in memory)
- Enqueues evaluation tasks; serves leaderboard queries (cached in Redis)

**Celery workers:**
- `validation` queue (fast, small tasks): schema/coverage checks
- `evaluation` queue (heavy): metric computation. Separate queues so a 20-minute segmentation job never blocks quick validations.
- Evaluators are plugins: one Python class per task type implementing `validate(file) -> Report` and `evaluate(preds, gt) -> Metrics`.

**PostgreSQL:** source of truth. **Redis:** Celery broker + result backend + leaderboard cache. **Object storage:** raw uploads (retention policy, e.g. 90 days for non-published) and encrypted ground truth.

### 5.2 Data model (core tables)

```
users(id, email, password_hash, name, affiliation, role, created_at)
api_tokens(id, user_id, token_hash, name, last_used_at)

benchmarks(id, slug, name, task_type, description, dataset_url, license,
           primary_metric, eval_config_yaml, evaluator_version, is_active)
ground_truths(id, benchmark_id, split, storage_uri, checksum, num_samples)

submissions(id, user_id, benchmark_id, status, file_uri, file_format,
            checksum, error_report_json, created_at, evaluated_at,
            evaluator_version, is_published)
results(id, submission_id, metric_name, value, per_class_json)

model_cards(id, submission_id, architecture, backbone, num_params,
            input_size, pretraining, external_data, hardware,
            training_time, repo_url, paper_url, notebook_url, notes)

leaderboard_entries  -- materialized view or cached query over
                     -- published submissions + best-per-user logic
```

### 5.3 Key API endpoints

```
POST   /auth/register | /auth/login | /auth/tokens
GET    /benchmarks                     GET /benchmarks/{slug}
GET    /benchmarks/{slug}/format-spec  GET /benchmarks/{slug}/leaderboard
POST   /benchmarks/{slug}/submissions        (returns presigned upload URL + submission id)
POST   /submissions/{id}/finalize            (client confirms upload done → triggers validation)
GET    /submissions/{id}                     (status + results + error report)
GET    /users/me/submissions
PUT    /submissions/{id}/model-card
POST   /submissions/{id}/publish
Admin: POST /admin/benchmarks, POST /admin/submissions/{id}/reevaluate, ...
```

### 5.4 Google Colab integration (your training side)

- **Starter notebooks** (one per benchmark) hosted in a public GitHub repo, opened via an "Open in Colab" badge on each benchmark page. Each notebook: download data → train baseline → inference → export → submit via API token.
- **`eobench` CLI/SDK** (thin `pip` package, ~300 lines): `download`, `validate` (runs the *same* validation code locally — saves quota-burning failed submissions), `submit`, `status`. Local validation reuses the evaluator plugin's `validate()` — ship it in the package.
- Colab practicalities baked into notebooks: mount Google Drive for checkpoints (Colab disconnects), keep datasets in Drive or stream from official sources, use free-tier-friendly batch sizes, resume-from-checkpoint cells.

---

## 6. Repository & Folder Structure

Monorepo (simplest to start; split later if needed):

```
eo-benchmark-hub/
├── docker-compose.yml              # dev: api, worker, redis, postgres, minio, frontend
├── docker-compose.prod.yml
├── .env.example
├── Makefile                        # make dev / test / lint / migrate / seed
├── README.md
│
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── alembic/                    # DB migrations
│   ├── app/
│   │   ├── main.py                 # FastAPI app factory
│   │   ├── config.py               # pydantic-settings
│   │   ├── api/
│   │   │   ├── deps.py             # auth deps, db session
│   │   │   └── routes/
│   │   │       ├── auth.py
│   │   │       ├── benchmarks.py
│   │   │       ├── submissions.py
│   │   │       ├── leaderboard.py
│   │   │       ├── model_cards.py
│   │   │       └── admin.py
│   │   ├── models/                 # SQLAlchemy models
│   │   ├── schemas/                # Pydantic request/response schemas
│   │   ├── services/               # business logic (submission_service.py, ...)
│   │   ├── storage/                # S3/MinIO client, presigned URLs
│   │   └── security/               # JWT, token hashing, rate limiting
│   └── tests/
│
├── evaluation/                     # shared by workers AND the eobench CLI
│   ├── pyproject.toml
│   ├── eobench_eval/
│   │   ├── registry.py             # task_type → evaluator plugin
│   │   ├── base.py                 # Evaluator ABC: validate(), evaluate()
│   │   ├── evaluators/
│   │   │   ├── scene_classification.py    # EuroSAT
│   │   │   ├── multilabel.py              # BigEarthNet
│   │   │   ├── segmentation.py            # LandCover.ai (rasterio, torchmetrics)
│   │   │   └── object_extraction.py       # SpaceNet (shapely, geopandas)
│   │   ├── io/                     # CSV/JSON/GeoTIFF/GeoJSON readers
│   │   └── metrics/                # thin wrappers over sklearn/torchmetrics
│   └── tests/                      # golden tests: known preds → known scores
│
├── worker/
│   ├── Dockerfile
│   ├── celery_app.py               # queues: validation, evaluation
│   └── tasks/
│       ├── validate_submission.py
│       └── evaluate_submission.py
│
├── cli/                            # the pip-installable `eobench` package
│   ├── pyproject.toml
│   └── eobench/
│       ├── cli.py                  # click/typer: download, validate, submit, status
│       ├── client.py               # API client
│       └── export.py               # helpers to write correct prediction files
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── main.tsx
│       ├── api/                    # generated client from OpenAPI spec
│       ├── components/
│       │   ├── leaderboard/        # table, filters, sparkline history
│       │   ├── submission/         # uploader (tus/resumable), status tracker
│       │   ├── modelcard/
│       │   └── common/
│       ├── pages/
│       │   ├── Home.tsx
│       │   ├── BenchmarkList.tsx
│       │   ├── BenchmarkDetail.tsx # spec, leaderboard, "Open in Colab" badge
│       │   ├── SubmitPage.tsx
│       │   ├── SubmissionDetail.tsx
│       │   ├── ModelCardPage.tsx
│       │   ├── Profile.tsx
│       │   └── Docs.tsx
│       ├── hooks/
│       └── store/                  # auth state, react-query cache
│
├── notebooks/                      # starter Colab notebooks (also linked from site)
│   ├── eurosat_baseline.ipynb
│   ├── bigearthnet_baseline.ipynb
│   ├── landcoverai_baseline.ipynb
│   └── spacenet_baseline.ipynb
│
├── infra/
│   ├── nginx/nginx.conf
│   └── deploy/                     # VM setup scripts or Terraform (later)
│
└── docs/
    ├── submission-formats.md       # exact schemas per benchmark
    ├── metrics.md                  # formal metric definitions
    └── api.md                      # or auto-generated OpenAPI
```

---

## 7. Task Breakdown & Milestones

No deadline (D10), so phases are ordered by **dependency**, each with an explicit exit criterion. Don't start a phase until the previous one's exit criterion passes — this keeps the project always in a working, demoable state.

### Phase 0 — Foundations
- [ ] Monorepo skeleton per §6; Docker Compose (postgres, redis, api, worker, frontend)
- [ ] CI (ruff, pytest, frontend lint/build), `.env.example`, Makefile (`make dev/test/migrate/seed`)
- [ ] Core tables + Alembic migrations
- [ ] Auth: register/login/JWT, hashed API tokens, role field (user/maintainer/admin)
- [ ] `Storage` interface + filesystem implementation
- **Exit:** `make dev` boots everything; you can register, log in, and create an API token via the OpenAPI docs page.

### Phase 1 — EuroSAT end-to-end ← the critical milestone
- [ ] `scripts/make_splits.py` + EuroSAT split manifest (§3.4); seed hidden labels
- [ ] Evaluator ABC + registry; scene-classification evaluator (CSV/JSON) with golden tests
- [ ] Submission flow: upload → validate (sync) → queue → evaluate (Celery) → results
- [ ] Minimal React: benchmark page, uploader, submission status, results view
- [ ] EuroSAT baseline Colab notebook (ResNet-18): train → infer → export CSV → upload via web UI
- **Exit:** a prediction file produced in Colab gets a real accuracy/F1 score on your local instance.

### Phase 2 — Multi-user leaderboard + model cards
- [ ] Model card form + publish flow (card required to publish)
- [ ] Public leaderboard: sorting, filtering, best-per-user collapsing, Redis caching
- [ ] Profiles, submission history, per-user weekly quota enforcement
- **Exit:** two different accounts submit; leaderboard shows each user's best, correctly ranked.

### Phase 3 — Remaining benchmarks (one at a time, each fully done before the next)
- [ ] BigEarthNet: multi-label evaluator (micro/macro F1, mAP) + splits + ResNet-18 notebook
- [ ] LandCover.ai: segmentation evaluator — ZIP of PNG masks, streamed/chunked mIoU so large ZIPs never load fully into RAM + small U-Net notebook
- [ ] SpaceNet: building-footprint evaluator (shapely IoU-thresholded polygon F1, GeoJSON input) + splits; baseline notebook optional
- **Exit per benchmark:** golden tests pass and one real Colab-produced submission scores correctly.

### Phase 4 — `eobench` CLI + docs
- [ ] `pip install eobench`: `download`, `validate` (reuses evaluator `validate()` locally), `submit`, `status`
- [ ] ngrok/cloudflared helper script for live Colab→API demos
- [ ] Docs pages: submission format specs, metric definitions, "how to submit from Colab"
- **Exit:** `eobench validate` catches a deliberately broken file locally; `eobench submit` works over an ngrok tunnel from Colab.

### Phase 5 — Admin, polish & demo
- [ ] Admin panel (SQLAdmin): users, benchmarks, submissions, re-evaluate button (versioned evaluators)
- [ ] "Prod-like" compose profile: `vite build` served by FastAPI StaticFiles, seeded demo data, one-command startup
- [ ] Flower for Celery, structured logging, optional Sentry
- [ ] Backup script for `./data/` (pg_dump + storage + ground truth); README with screenshots
- **Exit:** fresh clone + one command = fully working seeded demo.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Test-set leakage / overfitting via many submissions | Hidden labels, weekly quota, optional public dev split |
| Huge segmentation uploads crash workers | Presigned direct-to-storage upload, streaming/chunked evaluation, per-task memory limits |
| Metric bugs destroy trust | Golden tests with hand-computed scores; versioned evaluators; re-run capability |
| Dataset licensing (BigEarthNet, SpaceNet have specific licenses) | Never redistribute data; link to official sources; check each license before hosting *any* derived labels |
| Colab free tier limits (session timeouts, GPU quotas) | Checkpoint-resume patterns in notebooks; small-model baselines |
| SpaceNet APLS metric complexity | Ship building-footprint F1 first; APLS in v2 |

---

## 9. Decisions — All Resolved

| # | Decision | Resolution | Consequence |
|---|---|---|---|
| D1 | Backend | **FastAPI** | Auto OpenAPI → generated TS client for React; SQLAlchemy + Alembic; SQLAdmin for the admin panel (Django-admin equivalent) |
| D2 | Hosting | **None — local Docker Compose** | Filesystem storage behind `Storage` interface; no NGINX/TLS; multipart uploads; Colab submits via web UI (ngrok optional for CLI demos) |
| D3 | First benchmark | **EuroSAT** | Smallest dataset, CSV format, trains in minutes on free Colab T4 — fastest path to end-to-end |
| D4 | Test labels | **Custom re-split** | Deterministic stratified splits, published manifest, hidden labels worker-only (§3.4) |
| D5 | Quota | **Config, default 20/week/benchmark** | Enforced per user — matters now that it's multi-user |
| D6 | Segmentation format | **ZIP of per-tile PNG masks** | No rasterio in the validation path; trivial Colab export; GeoTIFF support can come in v2 |
| D7 | Auth | **Email + password, no verification** | Argon2/bcrypt hashing, JWT sessions, hashed API tokens — real auth, since multiple users share the instance |
| D8 | Users | **Multi-user** | Full account system, per-user submission history, best-per-user leaderboard logic, quotas, admin moderation all stay in scope |
| D9 | Colab baselines | **One tiny baseline per benchmark** | ResNet-18 (EuroSAT, BigEarthNet), small U-Net (LandCover.ai); SpaceNet baseline optional/last |
| D10 | Timeline | **No deadline — full scope** | Nothing cut: all four benchmarks, `eobench` CLI, admin panel, docs. Phases ordered by dependency, not weeks |
| D11 | Repo layout | **Monorepo** | As per §6 |
| D12 | Retention | **Keep everything** | Local disk; backup script covers `./data/` |

**Multi-user note (D8):** "multi-user, not hosted" means the instance runs on your machine/LAN and others reach it over your network (or a temporary ngrok tunnel). Auth is therefore built properly — hashed passwords, per-user tokens, role checks — so nothing needs rework if you ever put it on a VPS. The only hosting-specific items deferred are TLS, email delivery, and cloud storage, each isolated behind an interface or config flag.
