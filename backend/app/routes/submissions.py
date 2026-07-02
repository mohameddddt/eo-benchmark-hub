from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.config import settings
from app.database import get_db
from app.deps import get_current_user
from app.routes.benchmarks import get_benchmark_or_404
from app.storage import storage
from worker.celery_app import celery_app

router = APIRouter(tags=["submissions"])

MAX_UPLOAD_BYTES = 2 * 1024**3  # 2 GB (segmentation ZIPs, later)
ALLOWED_FORMATS = {"csv", "json", "zip", "geojson"}


def to_out(sub: models.Submission) -> schemas.SubmissionOut:
    return schemas.SubmissionOut(
        id=sub.id,
        status=sub.status,
        file_format=sub.file_format,
        original_filename=sub.original_filename,
        checksum=sub.checksum,
        error_report=sub.error_report,
        evaluator_version=sub.evaluator_version,
        is_published=sub.is_published,
        created_at=sub.created_at,
        evaluated_at=sub.evaluated_at,
        benchmark_slug=sub.benchmark.slug,
        results=[schemas.ResultOut.model_validate(r) for r in sub.results],
    )


@router.post("/benchmarks/{slug}/submissions", response_model=schemas.SubmissionOut, status_code=201)
def create_submission(
    slug: str,
    file: UploadFile = File(...),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    bm = get_benchmark_or_404(slug, db)

    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_FORMATS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Unsupported file type '.{ext}'. Allowed: {sorted(ALLOWED_FORMATS)}",
        )

    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    recent = (
        db.query(models.Submission)
        .filter(
            models.Submission.user_id == user.id,
            models.Submission.benchmark_id == bm.id,
            models.Submission.created_at >= week_ago,
        )
        .count()
    )
    if recent >= settings.weekly_quota:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Weekly quota reached ({settings.weekly_quota} submissions per benchmark). Try again later.",
        )

    sub = models.Submission(
        user_id=user.id,
        benchmark_id=bm.id,
        file_format=ext,
        original_filename=file.filename or "upload",
        file_uri="",
        status="UPLOADED",
    )
    db.add(sub)
    db.flush()  # get sub.id before saving the file under it

    key = f"submissions/{bm.slug}/{sub.id}.{ext}"
    uri, checksum = storage.save(key, file.file)
    sub.file_uri, sub.checksum, sub.status = uri, checksum, "QUEUED"
    db.commit()
    db.refresh(sub)

    celery_app.send_task("worker.tasks.process_submission", args=[sub.id], queue="evaluation")
    return to_out(sub)


@router.get("/submissions/{submission_id}", response_model=schemas.SubmissionOut)
def get_submission(
    submission_id: str,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sub = db.get(models.Submission, submission_id)
    if sub is None or (sub.user_id != user.id and not sub.is_published and user.role != "admin"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Submission not found")
    return to_out(sub)


@router.get("/users/me/submissions", response_model=list[schemas.SubmissionOut])
def my_submissions(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    subs = (
        db.query(models.Submission)
        .filter(models.Submission.user_id == user.id)
        .order_by(models.Submission.created_at.desc())
        .all()
    )
    return [to_out(s) for s in subs]


@router.put("/submissions/{submission_id}/model-card", response_model=schemas.SubmissionOut)
def upsert_model_card(
    submission_id: str,
    body: schemas.ModelCardIn,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sub = db.get(models.Submission, submission_id)
    if sub is None or sub.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Submission not found")
    if sub.model_card is None:
        sub.model_card = models.ModelCard(submission_id=sub.id)
        db.add(sub.model_card)
    for field, value in body.model_dump().items():
        setattr(sub.model_card, field, value)
    db.commit()
    db.refresh(sub)
    return to_out(sub)


@router.post("/submissions/{submission_id}/publish", response_model=schemas.SubmissionOut)
def publish(
    submission_id: str,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sub = db.get(models.Submission, submission_id)
    if sub is None or sub.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Submission not found")
    if sub.status != "COMPLETED":
        raise HTTPException(status.HTTP_409_CONFLICT, "Only completed submissions can be published")
    if sub.model_card is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Add a model card before publishing")
    sub.is_published = True
    db.commit()
    db.refresh(sub)
    return to_out(sub)
