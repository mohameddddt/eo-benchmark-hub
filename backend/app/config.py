from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://eobench:eobench-dev-password@localhost:5432/eobench"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str = "dev-secret"
    jwt_expire_minutes: int = 1440
    storage_dir: str = "/data/storage"
    ground_truth_dir: str = "/data/ground_truth"
    auto_create_tables: bool = True
    weekly_quota: int = 20


settings = Settings()
