"""Dispatch evaluation jobs to the configured backend."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from app.config import settings
from app.services.evaluation import run_submission

log = logging.getLogger("eobench.runner")
_executor: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        # one worker: evaluations are CPU-bound and fast; ordering stays FIFO
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eobench-eval")
    return _executor


def enqueue(submission_id: str) -> None:
    backend = settings.eval_backend
    if backend == "sync":
        run_submission(submission_id)
    elif backend == "celery":
        from worker.celery_app import celery_app

        celery_app.send_task("worker.tasks.process_submission", args=[submission_id], queue="evaluation")
    else:
        _pool().submit(run_submission, submission_id)


def recover_pending() -> None:
    """Inline mode loses its queue on restart — re-enqueue anything unfinished."""
    if settings.eval_backend != "inline":
        return
    from app.database import SessionLocal
    from app.models import RUNNING_STATUSES, Submission

    db = SessionLocal()
    try:
        ids = [s.id for s in db.query(Submission).filter(Submission.status.in_(RUNNING_STATUSES))]
    finally:
        db.close()
    for sid in ids:
        log.info("re-enqueueing unfinished submission %s", sid)
        enqueue(sid)
