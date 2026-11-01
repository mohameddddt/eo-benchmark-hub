# EO-Bench Hub — common tasks. Python commands use .venv if present.
PY ?= $(shell [ -x .venv/Scripts/python ] && echo .venv/Scripts/python || ([ -x .venv/bin/python ] && echo .venv/bin/python || echo python))
export PYTHONPATH := backend:evaluation:.

.PHONY: setup dev api web serve seed test lint kaggle-push kaggle-pull import static compose

setup:                ## install backend + frontend dependencies
	$(PY) -m pip install -r backend/requirements-dev.txt -e evaluation
	cd frontend && npm install

api:                  ## API with auto-reload on :8000
	$(PY) -m uvicorn app.main:app --app-dir backend --reload --port 8000

web:                  ## Vite dev server on :5173 (proxies /api to :8000)
	cd frontend && npm run dev

serve:                ## production-like: build the SPA, serve everything from :8000
	cd frontend && npm run build
	$(PY) -m uvicorn app.main:app --app-dir backend --port 8000

seed:                 ## register benchmarks, score Kaggle baselines
	$(PY) scripts/seed.py

test:
	$(PY) -m pytest backend/tests evaluation/tests -q

lint:
	$(PY) -m ruff check backend evaluation worker scripts training

kaggle-push:          ## make kaggle-push B=eurosat
	$(PY) training/kaggle.py push $(B)

kaggle-pull:          ## make kaggle-pull B=eurosat && make import B=eurosat
	$(PY) training/kaggle.py pull $(B)

import:
	$(PY) scripts/import_kaggle.py $(B)

static:               ## snapshot the API and build the read-only showcase
	$(PY) scripts/export_static.py
	cd frontend && npm run build:static

compose:              ## Postgres + Redis + Celery stack
	docker compose up --build
