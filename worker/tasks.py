"""Submission processing pipeline: VALIDATING -> EVALUATING -> COMPLETED.

Failure semantics:
  - validation errors  -> REJECTED (user's fault, human-readable error_report)
  - evaluator crashes  -> FAILED   (our fault, retried up to 3 times)
"""
import traceback
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.database import SessionLocal
from app.models import Result, Submission
from app.storage import storage
from eobench_eval import get_evaluator
from worker.celery_app import celery_app


@celery_app.task(name="worker.tasks.process_submission", bind=True, max_retries=3)
def process_submission(self, submission_id: str) -> None:
    db = SessionLocal()
    try:
        sub = db.get(Submission, submission_id)
        if sub is None:
            return
        bm = sub.benchmark
        config = {"classes": bm.classes or [], "split_version": bm.split_version}
        evaluator = get_evaluator(bm.task_type)
        pred_path = storage.path(sub.file_uri)
        gt_dir = Path(settings.ground_truth_dir) / bm.slug

        sub.status = "VALIDATING"
        db.commit()
        gt = evaluator.load_ground_truth(gt_dir, config)
        errors = evaluator.validate(pred_path, gt, config)
        if errors:
            sub.status = "REJECTED"
            sub.error_report = {"errors": errors}
            sub.evaluated_at = datetime.now(timezone.utc)
            db.commit()
            return

        sub.status = "EVALUATING"
        db.commit()
        outcome = evaluator.evaluate(pred_path, gt, config)

        db.query(Result).filter(Result.submission_id == sub.id).delete()
        metrics = outcome["metrics"]
        for name, value in metrics.items():
            db.add(
                Result(
                    submission_id=sub.id,
                    metric_name=name,
                    value=value,
                    # attach the detail payload once, on the primary metric row
                    extra=outcome.get("extra") if name == bm.primary_metric else None,
                )
            )
        sub.status = "COMPLETED"
        sub.error_report = None
        sub.evaluator_version = evaluator.version
        sub.evaluated_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as exc:
        db.rollback()
        sub = db.get(Submission, submission_id)
        if sub is not None:
            sub.status = "FAILED"
            sub.error_report = {
                "errors": [f"Internal evaluation error: {exc}"],
                "traceback": traceback.format_exc()[-2000:],
            }
            sub.evaluated_at = datetime.now(timezone.utc)
            db.commit()
        raise self.retry(exc=exc, countdown=10)
    finally:
        db.close()
