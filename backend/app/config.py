"""Runtime settings (environment variables, case-insensitive).

Defaults are chosen so that `uvicorn app.main:app` works from a fresh clone
with nothing else installed: SQLite under ./data, evaluation in a background
thread. docker-compose overrides them for the Postgres + Redis + Celery stack.
"""
from pathlib import Path

from pydantic_settings import BaseSettings

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"


class Settings(BaseSettings):
    database_url: str = f"sqlite:///{(DATA / 'eobench.db').as_posix()}"
    # inline: background thread in the API process · celery: Redis-backed workers ·
    # sync: evaluate inside the request (tests, seeding)
    eval_backend: str = "inline"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str = "dev-secret-change-me"
    jwt_expire_minutes: int = 1440
    storage_dir: str = str(DATA / "storage")
    ground_truth_dir: str = str(DATA / "ground_truth")
    frontend_dist: str = str(REPO / "frontend" / "dist")
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    auto_create_tables: bool = True
    weekly_quota: int = 20
    max_upload_mb: int = 512


settings = Settings()
