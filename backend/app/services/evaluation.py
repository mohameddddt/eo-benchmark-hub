"""The submission pipeline: VALIDATING -> EVALUATING -> COMPLETED.

Shared by every execution backend (inline thread, Celery worker, sync).

Failure semantics:
  - validation errors / SubmissionError -> REJECTED (user's fault, readable report)
  - missing ground truth                -> FAILED   (server misconfigured)
  - evaluator crash                     -> FAILED   (our fault; traceback kept for admins)
Evaluators are deterministic, so a crash is not retried — an admin re-runs it
after the fix (POST /api/admin/submissions/{id}/reevaluate).
"""
from __future__ import annotations

import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.database import SessionLocal
from app.models import Benchmark, Result, Submission
from app.storage import storage
from eobench_eval import get_evaluator
from eobench_eval.base import SubmissionError

log = logging.getLogger("eobench.eval")


def benchmark_config(bm: Benchmark) -> dict:
    return {**(bm.config or {}), "classes": bm.classes or [], "split_version": bm.split_version}


def gt_dir(bm: Benchmark) -> Path:
    """Benchmarks may share ground truth (EuroSAT-C reuses EuroSAT's labels)."""
    return Path(settings.ground_truth_dir) / ((bm.config or {}).get("ground_truth") or bm.slug)


def is_ready(bm: Benchmark) -> bool:
    try:
        ev = get_evaluator(bm.task_type)
    except KeyError:
        return False
    d = gt_dir(bm)
    return all((d / f).exists() for f in ev.required_files(benchmark_config(bm)))


def _finish(db, sub: Submission, status: str, errors: list[str] | None = None, **extra) -> None:
    sub.status = status
    sub.error_report = {"errors": errors, **extra} if errors else None
    sub.evaluated_at = datetime.now(UTC)
    db.commit()


def run_submission(submission_id: str) -> str:
    """Evaluate one submission end to end. Never raises; returns the final status."""
    db = SessionLocal()
    try:
        sub = db.get(Submission, submission_id)
        if sub is None:
            return "MISSING"
        bm = sub.benchmark
        config = benchmark_config(bm)
        evaluator = get_evaluator(bm.task_type)
        pred_path = storage.path(sub.file_uri)

        sub.status = "VALIDATING"
        db.commit()
        try:
            gt = evaluator.load_ground_truth(gt_dir(bm), config)
        except FileNotFoundError:
            _finish(db, sub, "FAILED", [f"Benchmark '{bm.slug}' has no hidden test split installed on "
                                        "this server, so this submission cannot be scored."])
            return sub.status
        errors = evaluator.validate(pred_path, gt, config)
        if errors:
            _finish(db, sub, "REJECTED", errors)
            return sub.status

        sub.status = "EVALUATING"
        db.commit()
        try:
            outcome = evaluator.evaluate(pred_path, gt, config)
        except SubmissionError as e:
            _finish(db, sub, "REJECTED", [str(e)])
            return sub.status

        db.query(Result).filter(Result.submission_id == sub.id).delete()
        ci = outcome.get("ci", {})
        for name, value in outcome["metrics"].items():
            lo, hi = ci.get(name, (None, None))
            db.add(Result(submission_id=sub.id, metric_name=name, value=value, ci_low=lo, ci_high=hi))
        sub.details = outcome.get("extra")
        sub.evaluator_version = evaluator.version
        _finish(db, sub, "COMPLETED")
        log.info("submission %s scored %s", sub.id, outcome["metrics"].get(bm.primary_metric))
        return sub.status
    except Exception as exc:  # noqa: BLE001 — the job must always land in a terminal state
        db.rollback()
        sub = db.get(Submission, submission_id)
        if sub is not None:
            _finish(db, sub, "FAILED", [f"Internal evaluation error: {exc}"],
                    traceback=traceback.format_exc()[-3000:])
        log.exception("submission %s failed", submission_id)
        return "FAILED"
    finally:
        db.close()
