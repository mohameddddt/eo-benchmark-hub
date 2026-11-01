"""Creating submissions and shaping them for the API."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import BinaryIO

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.config import settings
from app.services import leaderboard
from app.services.evaluation import is_ready
from app.storage import storage
from eobench_eval import get_evaluator


def quota_used(db: Session, user: models.User, bm: models.Benchmark) -> int:
    """Rejected uploads are free: they reveal nothing about the hidden labels."""
    week_ago = datetime.now(UTC) - timedelta(days=7)
    return (
        db.query(models.Submission)
        .filter(models.Submission.user_id == user.id, models.Submission.benchmark_id == bm.id,
                models.Submission.created_at >= week_ago, models.Submission.status != "REJECTED")
        .count()
    )


def create(db: Session, user: models.User, bm: models.Benchmark, stream: BinaryIO, filename: str,
           model_name: str, description: str = "", enforce_quota: bool = True) -> models.Submission:
    if not bm.is_active or not is_ready(bm):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"Benchmark '{bm.slug}' is not accepting submissions yet — its hidden test "
                            "split is not installed on this server.")
    ext = (filename or "").rsplit(".", 1)[-1].lower()
    allowed = get_evaluator(bm.task_type).formats
    if ext not in allowed:
        raise HTTPException(422,
                            f"Unsupported file type '.{ext}' for {bm.name}. Expected: "
                            + ", ".join(f".{a}" for a in allowed))
    if enforce_quota and quota_used(db, user, bm) >= settings.weekly_quota:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f"Weekly quota reached ({settings.weekly_quota} scored submissions per "
                            "benchmark). Rejected uploads don't count — try again later.")

    sub = models.Submission(user_id=user.id, benchmark_id=bm.id, file_format=ext,
                            original_filename=filename or "upload", file_uri="", status="UPLOADED",
                            model_name=model_name.strip()[:120] or "Unnamed model",
                            description=description[:2000])
    db.add(sub)
    db.flush()
    key = f"submissions/{bm.slug}/{sub.id}.{ext}"
    uri, checksum, size = storage.save(key, stream, max_bytes=settings.max_upload_mb * 1024 ** 2)
    if uri is None:
        db.rollback()
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"File exceeds the {settings.max_upload_mb} MB upload limit")
    sub.file_uri, sub.checksum, sub.file_size, sub.status = uri, checksum, size, "QUEUED"
    db.commit()
    db.refresh(sub)
    return sub


def to_out(db: Session, sub: models.Submission, with_rank: bool = True) -> schemas.SubmissionOut:
    bm = sub.benchmark
    rank = n_ranked = None
    if with_rank and sub.is_published and sub.status == "COMPLETED":
        board = leaderboard.build(db, bm, collapse=False)
        n_ranked = len(board)
        rank = next((e.rank for e in board if e.submission_id == sub.id), None)
    return schemas.SubmissionOut(
        id=sub.id, status=sub.status,
        benchmark=schemas.BenchmarkRef(slug=bm.slug, name=bm.name, task_type=bm.task_type,
                                       primary_metric=bm.primary_metric),
        user=schemas.UserRef.model_validate(sub.user),
        model_name=sub.model_name, description=sub.description, file_format=sub.file_format,
        file_size=sub.file_size, original_filename=sub.original_filename, checksum=sub.checksum,
        error_report=sub.error_report, evaluator_version=sub.evaluator_version,
        is_published=sub.is_published, verified=sub.verified, created_at=sub.created_at,
        evaluated_at=sub.evaluated_at,
        results=[schemas.ResultOut.model_validate(r) for r in sub.results],
        details=sub.details,
        model_card=schemas.ModelCardOut.model_validate(sub.model_card) if sub.model_card else None,
        rank=rank, n_ranked=n_ranked,
    )
