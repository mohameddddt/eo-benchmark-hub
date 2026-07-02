from celery import Celery

from app.config import settings

celery_app = Celery("eobench", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_default_queue="evaluation",
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # heavy jobs: one at a time per process
)
celery_app.autodiscover_tasks(["worker"])
