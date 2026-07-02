.PHONY: dev down logs seed test psql lint

dev:            ## build & start everything (API :8000, frontend :5173)
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f api worker

seed:           ## register the EuroSAT benchmark (+ ground truth check)
	docker compose exec api python /srv/scripts/seed.py

test:
	docker compose exec api pytest /srv/backend/tests /srv/evaluation/tests -q

psql:
	docker compose exec postgres psql -U $${POSTGRES_USER:-eobench} $${POSTGRES_DB:-eobench}

lint:
	docker compose exec api ruff check /srv/backend /srv/evaluation /srv/worker
