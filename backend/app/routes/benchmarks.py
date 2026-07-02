from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


def get_benchmark_or_404(slug: str, db: Session) -> models.Benchmark:
    bm = db.query(models.Benchmark).filter(models.Benchmark.slug == slug).first()
    if bm is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Benchmark '{slug}' not found")
    return bm


@router.get("", response_model=list[schemas.BenchmarkOut])
def list_benchmarks(db: Session = Depends(get_db)):
    return db.query(models.Benchmark).filter(models.Benchmark.is_active).all()


@router.get("/{slug}", response_model=schemas.BenchmarkOut)
def get_benchmark(slug: str, db: Session = Depends(get_db)):
    return get_benchmark_or_404(slug, db)


@router.get("/{slug}/leaderboard", response_model=list[schemas.LeaderboardEntry])
def leaderboard(slug: str, db: Session = Depends(get_db)):
    bm = get_benchmark_or_404(slug, db)
    rows = (
        db.query(models.Result, models.Submission, models.User)
        .join(models.Submission, models.Result.submission_id == models.Submission.id)
        .join(models.User, models.Submission.user_id == models.User.id)
        .filter(
            models.Submission.benchmark_id == bm.id,
            models.Submission.status == "COMPLETED",
            models.Submission.is_published,
            models.Result.metric_name == bm.primary_metric,
        )
        .order_by(models.Result.value.desc())
        .all()
    )
    best_per_user: dict[str, tuple] = {}
    for result, sub, user in rows:
        if user.id not in best_per_user:  # rows already sorted desc — first hit is the best
            best_per_user[user.id] = (result, sub, user)
    entries = sorted(best_per_user.values(), key=lambda t: t[0].value, reverse=True)
    return [
        schemas.LeaderboardEntry(
            rank=i + 1,
            user_name=user.name,
            submission_id=sub.id,
            primary_metric=bm.primary_metric,
            score=result.value,
            created_at=sub.created_at,
        )
        for i, (result, sub, user) in enumerate(entries)
    ]
