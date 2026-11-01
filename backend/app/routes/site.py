"""Public, read-only aggregates: landing-page stats and researcher profiles."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db
from app.routes.benchmarks import _metric_label
from app.services import leaderboard

router = APIRouter(tags=["site"])


@router.get("/stats", response_model=schemas.StatsOut)
def stats(db: Session = Depends(get_db)):
    bms = db.query(models.Benchmark).filter(models.Benchmark.is_active).order_by(models.Benchmark.position).all()
    leaders, recent_pool = [], []
    for bm in bms:
        board = leaderboard.build(db, bm)
        label = _metric_label(bm)
        if board:
            gap = board[0].score - board[1].score if len(board) > 1 else None
            leaders.append(schemas.LeaderHighlight(
                benchmark_slug=bm.slug, benchmark_name=bm.name, metric_label=label,
                model_name=board[0].model_name, submission_id=board[0].submission_id,
                score=board[0].score, runner_up_gap=gap))
        recent_pool += [(e, bm, label) for e in board]
    published = (db.query(models.Submission)
                 .filter(models.Submission.is_published, models.Submission.status == "COMPLETED"))
    evaluated_at = {s.id: s.evaluated_at for s in published}
    recent_pool.sort(key=lambda t: evaluated_at.get(t[0].submission_id) or t[0].created_at, reverse=True)
    return schemas.StatsOut(
        benchmarks=len(bms),
        task_types=len({b.task_type for b in bms}),
        models=published.with_entities(func.count(func.distinct(models.Submission.model_name))).scalar() or 0,
        evaluations=db.query(models.Submission).filter(models.Submission.status == "COMPLETED").count(),
        researchers=published.with_entities(func.count(func.distinct(models.Submission.user_id))).scalar() or 0,
        leaders=leaders,
        recent=[schemas.RecentEntry(
            submission_id=e.submission_id, model_name=e.model_name, benchmark_slug=bm.slug,
            benchmark_name=bm.name, user_name=e.user.name, metric_label=label, score=e.score,
            evaluated_at=evaluated_at.get(e.submission_id)) for e, bm, label in recent_pool[:8]],
    )


@router.get("/users/{user_id}", response_model=schemas.UserPublic)
def public_profile(user_id: str, db: Session = Depends(get_db)):
    user = db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    entries = []
    for bm in db.query(models.Benchmark).filter(models.Benchmark.is_active).order_by(models.Benchmark.position):
        board = leaderboard.build(db, bm)
        for e in board:
            if e.user.id == user.id:
                entries.append({"benchmark_slug": bm.slug, "benchmark_name": bm.name,
                                "metric_label": _metric_label(bm), "rank": e.rank, "of": len(board),
                                "model_name": e.model_name, "score": e.score,
                                "submission_id": e.submission_id, "verified": e.verified})
    return schemas.UserPublic(id=user.id, name=user.name, affiliation=user.affiliation,
                              website=user.website, bio=user.bio, created_at=user.created_at,
                              entries=entries)
