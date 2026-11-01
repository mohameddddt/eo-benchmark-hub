"""Leaderboard construction: collapsing, ranking and CI-aware tiers.

Collapsing: one row per (researcher, model name) — the best of its repeated
submissions. Distinct models from the same account all appear (a lab's model
zoo is not one entry), but resubmitting the same model to probe the test set
does not stack the board.

Tiers: the tier leader's 95% interval opens a tier; every following entry
whose interval reaches into it joins that tier. Ranks still order by point
estimate, but entries in one tier are not distinguishable at this test-set
size, and the UI says so.
"""
from __future__ import annotations

from sqlalchemy.orm import Session, joinedload

from app import models, schemas
from eobench_eval import get_evaluator


def higher_is_better(bm: models.Benchmark, metric: str) -> bool:
    try:
        return get_evaluator(bm.task_type).metrics.get(metric, {}).get("higher_is_better", True)
    except KeyError:
        return True


def build(db: Session, bm: models.Benchmark, collapse: bool = True) -> list[schemas.LeaderboardEntry]:
    subs = (
        db.query(models.Submission)
        .options(joinedload(models.Submission.results), joinedload(models.Submission.user),
                 joinedload(models.Submission.model_card))
        .filter(models.Submission.benchmark_id == bm.id, models.Submission.status == "COMPLETED",
                models.Submission.is_published)
        .all()
    )
    hib = higher_is_better(bm, bm.primary_metric)
    sign = 1 if hib else -1
    rows = []
    for s in subs:
        prim = next((r for r in s.results if r.metric_name == bm.primary_metric), None)
        if prim is None:
            continue
        rows.append((s, prim))
    rows.sort(key=lambda t: (-sign * t[1].value, t[0].created_at))

    if collapse:
        seen, kept = set(), []
        for s, prim in rows:
            key = (s.user_id, (s.model_name or s.id).strip().lower())
            if key not in seen:
                seen.add(key)
                kept.append((s, prim))
        rows = kept

    out: list[schemas.LeaderboardEntry] = []
    tier, tier_bound = 0, None
    for i, (s, prim) in enumerate(rows):
        lo = prim.ci_low if prim.ci_low is not None else prim.value
        hi = prim.ci_high if prim.ci_high is not None else prim.value
        reach = hi if hib else lo            # how far this entry's interval reaches toward the top
        joins = tier_bound is not None and ((reach >= tier_bound) if hib else (reach <= tier_bound))
        if not joins:
            tier += 1
            tier_bound = lo if hib else hi   # the new leader's opposite bound opens the tier
        card = s.model_card
        out.append(schemas.LeaderboardEntry(
            rank=i + 1, tier=tier, submission_id=s.id, model_name=s.model_name or "Unnamed model",
            user=schemas.UserRef.model_validate(s.user), verified=s.verified, created_at=s.created_at,
            score=prim.value, ci_low=prim.ci_low, ci_high=prim.ci_high,
            metrics={r.metric_name: r.value for r in s.results},
            card=schemas.LeaderboardCard(
                architecture=card.architecture, family=card.family, pretraining=card.pretraining,
                num_params=card.num_params, gflops=card.gflops, throughput=card.throughput,
                input_size=card.input_size, external_data=card.external_data,
                notebook_url=card.notebook_url) if card else None,
        ))
    return out
