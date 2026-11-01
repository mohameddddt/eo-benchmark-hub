from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.config import settings
from app.database import get_db
from app.services import leaderboard
from app.services.evaluation import benchmark_config, is_ready
from eobench_eval import get_evaluator

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


def get_benchmark_or_404(slug: str, db: Session) -> models.Benchmark:
    bm = db.query(models.Benchmark).filter(models.Benchmark.slug == slug).first()
    if bm is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Benchmark '{slug}' not found")
    return bm


def _metric_label(bm: models.Benchmark) -> str:
    try:
        return get_evaluator(bm.task_type).metrics.get(bm.primary_metric, {}).get("label", bm.primary_metric)
    except KeyError:
        return bm.primary_metric


def summary(db: Session, bm: models.Benchmark) -> dict:
    board = leaderboard.build(db, bm)
    top = board[0] if board else None
    return dict(
        slug=bm.slug, name=bm.name, tagline=bm.tagline, task_type=bm.task_type, modality=bm.modality,
        primary_metric=bm.primary_metric, primary_metric_label=_metric_label(bm),
        n_classes=len(bm.classes or []), n_entries=len(board), ready=is_ready(bm),
        top=schemas.TopEntry(submission_id=top.submission_id, model_name=top.model_name,
                             score=top.score) if top else None,
        position=bm.position,
    )


@router.get("", response_model=list[schemas.BenchmarkSummary])
def list_benchmarks(db: Session = Depends(get_db)):
    bms = db.query(models.Benchmark).filter(models.Benchmark.is_active).order_by(models.Benchmark.position)
    return [schemas.BenchmarkSummary(**summary(db, b)) for b in bms]


@router.get("/{slug}", response_model=schemas.BenchmarkDetail)
def get_benchmark(slug: str, db: Session = Depends(get_db)):
    bm = get_benchmark_or_404(slug, db)
    ev = get_evaluator(bm.task_type)
    return schemas.BenchmarkDetail(
        **summary(db, bm),
        description=bm.description, dataset_url=bm.dataset_url, paper_url=bm.paper_url,
        starter_url=bm.starter_url, citation=bm.citation, license=bm.license, classes=bm.classes,
        config=bm.config, stats=bm.stats, split_version=bm.split_version,
        evaluator_version=ev.version,
        metrics=[schemas.MetricInfo(name=k, **v) for k, v in ev.metrics.items()],
        format_spec=ev.format_spec({**benchmark_config(bm), "id_prefix": (bm.config or {}).get("id_prefix", "xx")}),
        weekly_quota=settings.weekly_quota,
    )


@router.get("/{slug}/leaderboard", response_model=list[schemas.LeaderboardEntry])
def get_leaderboard(slug: str, all_submissions: bool = Query(False, alias="all"), db: Session = Depends(get_db)):
    """Published, scored entries. By default one row per (researcher, model);
    ?all=true lists every published submission."""
    return leaderboard.build(db, get_benchmark_or_404(slug, db), collapse=not all_submissions)
