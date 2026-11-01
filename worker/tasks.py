"""Celery entry point (EVAL_BACKEND=celery). The pipeline itself lives in
app.services.evaluation so the inline, sync and Celery backends share it."""
from app.services.evaluation import run_submission
from worker.celery_app import celery_app


@celery_app.task(name="worker.tasks.process_submission", acks_late=True)
def process_submission(submission_id: str) -> str:
    return run_submission(submission_id)
