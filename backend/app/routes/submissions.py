from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db
from app.deps import get_current_user, get_optional_user, require_maintainer
from app.routes.benchmarks import get_benchmark_or_404
from app.services import runner
from app.services import submissions as svc
from app.storage import storage

router = APIRouter(tags=["submissions"])


def _own(db: Session, submission_id: str, user: models.User) -> models.Submission:
    sub = db.get(models.Submission, submission_id)
    if sub is None or sub.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Submission not found")
    return sub


@router.post("/benchmarks/{slug}/submissions", response_model=schemas.SubmissionOut, status_code=201)
def create_submission(
    slug: str,
    file: UploadFile = File(...),
    model_name: str = Form("", max_length=120),
    description: str = Form("", max_length=2000),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    bm = get_benchmark_or_404(slug, db)
    sub = svc.create(db, user, bm, file.file, file.filename or "upload",
                     model_name or (file.filename or "upload").rsplit(".", 1)[0], description)
    runner.enqueue(sub.id)
    db.refresh(sub)
    return svc.to_out(db, sub)


@router.get("/submissions/{submission_id}", response_model=schemas.SubmissionOut)
def get_submission(submission_id: str, user: models.User | None = Depends(get_optional_user),
                   db: Session = Depends(get_db)):
    sub = db.get(models.Submission, submission_id)
    allowed = sub is not None and (sub.is_published or (user is not None and (
        sub.user_id == user.id or user.role in ("maintainer", "admin"))))
    if not allowed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Submission not found")
    out = svc.to_out(db, sub)
    if user is None or (sub.user_id != user.id and user.role not in ("maintainer", "admin")):
        if out.error_report:
            out.error_report.pop("traceback", None)
    return out


@router.get("/users/me/submissions", response_model=list[schemas.SubmissionOut])
def my_submissions(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    subs = (db.query(models.Submission).filter(models.Submission.user_id == user.id)
            .order_by(models.Submission.created_at.desc()).all())
    return [svc.to_out(db, s, with_rank=False) for s in subs]


@router.patch("/submissions/{submission_id}", response_model=schemas.SubmissionOut)
def update_submission(submission_id: str, body: schemas.SubmissionUpdateIn,
                      user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sub = _own(db, submission_id, user)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(sub, field, value)
    db.commit()
    return svc.to_out(db, sub)


@router.put("/submissions/{submission_id}/model-card", response_model=schemas.SubmissionOut)
def upsert_model_card(submission_id: str, body: schemas.ModelCardIn,
                      user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sub = _own(db, submission_id, user)
    if sub.model_card is None:
        sub.model_card = models.ModelCard(submission_id=sub.id)
        db.add(sub.model_card)
    for field, value in body.model_dump().items():
        setattr(sub.model_card, field, value)
    db.commit()
    db.refresh(sub)
    return svc.to_out(db, sub)


@router.post("/submissions/{submission_id}/publish", response_model=schemas.SubmissionOut)
def publish(submission_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sub = _own(db, submission_id, user)
    if sub.status != "COMPLETED":
        raise HTTPException(status.HTTP_409_CONFLICT, "Only completed submissions can be published")
    card = sub.model_card
    if card is None or not card.architecture.strip():
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "Add a model card (at least the architecture) before publishing")
    sub.is_published = True
    db.commit()
    return svc.to_out(db, sub)


@router.post("/submissions/{submission_id}/unpublish", response_model=schemas.SubmissionOut)
def unpublish(submission_id: str, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sub = _own(db, submission_id, user)
    sub.is_published = False
    db.commit()
    return svc.to_out(db, sub)


@router.delete("/submissions/{submission_id}", status_code=204)
def delete_submission(submission_id: str, user: models.User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    sub = _own(db, submission_id, user)
    if sub.is_published:
        raise HTTPException(status.HTTP_409_CONFLICT, "Unpublish the submission before deleting it")
    if sub.status in models.RUNNING_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, "Wait for evaluation to finish before deleting")
    storage.delete(sub.file_uri)
    db.delete(sub)
    db.commit()


# ---- maintainers ----
@router.post("/admin/submissions/{submission_id}/reevaluate", response_model=schemas.SubmissionOut,
             tags=["admin"])
def reevaluate(submission_id: str, _: models.User = Depends(require_maintainer), db: Session = Depends(get_db)):
    """Re-score with the current evaluator version (e.g. after a metric fix)."""
    sub = db.get(models.Submission, submission_id)
    if sub is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Submission not found")
    sub.status = "QUEUED"
    db.commit()
    runner.enqueue(sub.id)
    db.refresh(sub)
    return svc.to_out(db, sub)


@router.post("/admin/submissions/{submission_id}/verify", response_model=schemas.SubmissionOut, tags=["admin"])
def verify(submission_id: str, verified: bool = True, _: models.User = Depends(require_maintainer),
           db: Session = Depends(get_db)):
    """Mark an entry as independently reproduced from its public notebook."""
    sub = db.get(models.Submission, submission_id)
    if sub is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Submission not found")
    sub.verified = verified
    db.commit()
    return svc.to_out(db, sub)
