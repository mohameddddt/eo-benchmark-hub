#!/usr/bin/env python3
"""Register the benchmarks and score the Kaggle-trained baselines.

    python scripts/seed.py            # idempotent: only missing baselines are scored
    python scripts/seed.py --rescore  # re-run every baseline through the evaluators

Baselines go through exactly the same path as a researcher's upload
(storage -> validate -> evaluate); nothing here writes a metric directly.
A benchmark is listed as accepting submissions only once its hidden ground
truth is installed (scripts/import_kaggle.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import secrets
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "evaluation"), str(ROOT)]

from app.config import settings  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.models import Benchmark, ModelCard, Submission, User  # noqa: E402
from app.security import hash_password  # noqa: E402
from app.services import submissions as svc  # noqa: E402
from app.services.evaluation import gt_dir, is_ready, run_submission  # noqa: E402
from eobench_eval import splits  # noqa: E402
from eobench_eval.corruptions import CORRUPTIONS, SEVERITIES  # noqa: E402

KAGGLE = "https://www.kaggle.com/code/mohameddddt/eobench-{}"
BASELINES = ROOT / "data" / "baselines"
ORG = {"email": "baselines@example.com", "name": "EO-Bench Baselines",
       "affiliation": "EO-Bench Hub maintainers",
       "bio": "Reference models trained by the hub maintainers on Kaggle T4 GPUs with one shared, "
              "documented recipe per benchmark. Every entry links to the notebook that produced it."}

BENCHMARKS = [
    dict(
        slug="eurosat", name="EuroSAT", position=0, task_type="scene_classification",
        tagline="Land-use classification from Sentinel-2",
        modality="Sentinel-2 L1C · RGB · 10 m · 64×64 px",
        description=(
            "27,000 labelled Sentinel-2 patches across 10 land-use and land-cover classes, sampled from "
            "34 European countries. The hub's split_v2 keeps the v1 partition (70/10/10/10, stratified "
            "per class, seed 42) but replaces filename ids — which leaked the label — with content "
            "hashes. Modern backbones cluster at 98–99%, so read the confidence intervals, calibration "
            "and the EuroSAT-C robustness track to tell them apart."),
        dataset_url="https://zenodo.org/records/7711810",
        paper_url="https://arxiv.org/abs/1709.00029",
        citation="Helber, Bischke, Dengel, Borth. EuroSAT: A Novel Dataset and Deep Learning Benchmark "
                 "for Land Use and Land Cover Classification. IEEE JSTARS, 2019.",
        license="MIT", primary_metric="accuracy", classes=splits.EUROSAT_CLASSES,
        config={"id_prefix": "es"},
    ),
    dict(
        slug="eurosat-c", name="EuroSAT-C", position=1, task_type="scene_classification_robustness",
        tagline="Robustness to sensor and atmospheric corruptions",
        modality="EuroSAT hidden split × 7 corruptions × 3 severities",
        description=(
            "The EuroSAT hidden images re-scored under 21 deterministic corruptions that mimic real "
            "failure modes of optical EO pipelines: sensor noise, defocus, thin haze, cloud occlusion, "
            "downlink compression, coarser ground sampling and band-to-band misregistration. Corrupted "
            "images are generated locally with eobench_eval.corruptions — no extra download — and the "
            "primary metric is mean corruption accuracy (mCA)."),
        dataset_url="https://zenodo.org/records/7711810",
        paper_url="https://arxiv.org/abs/1903.12261",
        citation="Corruption protocol after Hendrycks & Dietterich, Benchmarking Neural Network "
                 "Robustness to Common Corruptions and Perturbations, ICLR 2019; EO-specific "
                 "corruptions by the EO-Bench Hub maintainers.",
        license="MIT", primary_metric="mca", classes=splits.EUROSAT_CLASSES,
        config={"id_prefix": "es", "ground_truth": "eurosat", "corruptions": list(CORRUPTIONS),
                "severities": list(SEVERITIES),
                "corruption_info": {k: {"description": v[0], "params": list(v[1])}
                                    for k, v in CORRUPTIONS.items()}},
    ),
    dict(
        slug="resisc45", name="NWPU-RESISC45", position=2, task_type="scene_classification",
        tagline="Fine-grained aerial scene recognition, 45 classes",
        modality="Google Earth aerial RGB · 0.2–30 m · 256×256 px",
        description=(
            "31,500 aerial images, 700 for each of 45 scene classes, with large variation in "
            "resolution, viewpoint, illumination and background — and several near-synonymous classes "
            "(dense vs medium vs sparse residential, church vs palace). split_v2 pools the Kaggle "
            "mirror's arbitrary train/test folders, drops exact byte-duplicates, and re-splits "
            "70/10/10/10 stratified per class."),
        dataset_url="https://www.kaggle.com/datasets/aqibrehmanpirzada/nwpuresisc45",
        paper_url="https://arxiv.org/abs/1703.00121",
        citation="Cheng, Han, Lu. Remote Sensing Image Scene Classification: Benchmark and State of the "
                 "Art. Proceedings of the IEEE, 2017.",
        license="Research use — see dataset terms", primary_metric="accuracy",
        classes=splits.RESISC45_CLASSES, config={"id_prefix": "rs"},
    ),
    dict(
        slug="landcoverai", name="LandCover.ai", position=3, task_type="semantic_segmentation",
        tagline="Mapping buildings, woodland, water and roads from orthophotos",
        modality="Polish aerial orthophotos · RGB · 25–50 cm · 512×512 tiles",
        description=(
            "41 aerial orthophotos covering 216 km² of rural Poland, hand-annotated for buildings, "
            "woodland, water and roads, cut into 512×512 tiles by the dataset's official split.py. "
            "The hub keeps the official train/val tiles and halves the official test tiles "
            "(seeded) into test_public and test_hidden. Classes are heavily imbalanced: buildings "
            "and roads are a few percent of pixels, so mIoU — not pixel accuracy — is primary."),
        dataset_url="https://landcover.ai.linuxpolska.com/",
        paper_url="https://arxiv.org/abs/2005.02264",
        citation="Boguszewski et al. LandCover.ai: Dataset for Automatic Mapping of Buildings, "
                 "Woodlands, Water and Roads from Aerial Imagery. CVPR Workshops, 2021.",
        license="CC BY-NC-SA 4.0", primary_metric="miou", classes=splits.LANDCOVERAI_CLASSES,
        config={"palette": [[222, 214, 190], [213, 62, 79], [46, 125, 79], [49, 116, 196], [245, 178, 40]]},
    ),
]


def manifest_stats(bm: Benchmark) -> dict | None:
    p = gt_dir(bm) / f"split_manifest_{bm.split_version}.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, newline="")))
    st = {"splits": dict(Counter(r["split"] for r in rows)), "fingerprint": splits.fingerprint(rows),
          "n_samples": len(rows)}
    if "label" in (rows[0] if rows else {}):
        st["class_counts"] = dict(Counter(r["label"] for r in rows))
    if bm.task_type == "scene_classification_robustness":
        n = st["splits"].get("test_hidden", 0)
        v = len(bm.config["corruptions"]) * len(bm.config["severities"])
        st["evaluated_images"] = n * (v + 1)
    zp = gt_dir(bm) / f"masks_hidden_{bm.split_version}.zip"
    if zp.exists():
        import numpy as np
        from PIL import Image

        counts = np.zeros(len(bm.classes), np.int64)
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                counts += np.bincount(np.asarray(Image.open(z.open(n))).ravel(), minlength=len(bm.classes))
        st["hidden_pixel_share"] = {c: float(v / counts.sum()) for c, v in zip(bm.classes, counts)}
    summ = BASELINES / bm.slug / "summary.json"
    if summ.exists():
        st["duplicates_dropped"] = json.loads(summ.read_text()).get("duplicates_dropped", 0)
    return st


def card_from_json(c: dict, bench: str) -> dict:
    hist = c.get("history") or []
    minutes = c.get("train_time_s", 0) / 60
    extra = {k: c.get(k) for k in ("lr", "batch_size", "weight_decay", "label_smoothing", "best_epoch",
                                   "val_acc", "val_miou", "test_public_acc", "test_public_miou", "norm",
                                   "loss", "arch", "encoder") if c.get(k) is not None}
    extra.update(c.get("extra") or {})
    env = c.get("environment", {})
    return dict(
        architecture=c.get("name", c["id"]),
        family=c.get("family", ""),
        backbone=c.get("encoder") or c.get("arch", ""),
        num_params=c.get("num_params"),
        gflops=c.get("gflops"),
        throughput=c.get("throughput_img_s"),
        throughput_setup=c.get("throughput_setup", ""),
        input_size=c.get("img_size"),
        pretraining=c.get("pretraining", ""),
        external_data=c.get("pretraining", "none") != "none",
        epochs=c.get("epochs"),
        optimizer=c.get("optimizer", ""),
        augmentation=c.get("augmentation", ""),
        hardware=f"Kaggle {env.get('gpu', 'GPU')}",
        training_time=f"{minutes:.0f} min" if minutes >= 1 else f"{minutes * 60:.0f} s",
        framework=f"PyTorch {env.get('torch', '').split('+')[0]}" + (f" · timm {env['timm']}" if env.get("timm") else ""),
        repo_url="https://github.com/mohameddddt/eo-benchmark-hub/tree/master/training",
        notebook_url=c.get("kernel_url") or KAGGLE.format(bench),
        notes=c.get("notes", ""),
        training_log=hist,
        extra=extra,
    )




def seed_baselines(db, org: User, bm: Benchmark, rescore: bool) -> None:
    src = BASELINES / bm.slug
    if not src.exists():
        return
    source_bench = (bm.config or {}).get("ground_truth") or bm.slug
    for mdir in sorted(p for p in src.iterdir() if p.is_dir()):
        card = json.loads((mdir / "card.json").read_text())
        pred = next((mdir / n for n in ("predictions.csv", "masks.zip") if (mdir / n).exists()), None)
        if pred is None:
            continue
        name = card.get("name", mdir.name)
        existing = (db.query(Submission).filter(Submission.user_id == org.id, Submission.benchmark_id == bm.id,
                                                Submission.model_name == name).first())
        if existing and existing.status == "COMPLETED" and not rescore:
            continue
        if existing:
            db.delete(existing)
            db.commit()
        with open(pred, "rb") as f:
            sub = svc.create(db, org, bm, f, pred.name, name,
                             description=f"Reference baseline · {card.get('pretraining', '')} pretraining",
                             enforce_quota=False)
        status = run_submission(sub.id)
        db.refresh(sub)
        if status != "COMPLETED":
            print(f"  !! {bm.slug}/{mdir.name}: {status} {sub.error_report}")
            continue
        sub.model_card = ModelCard(submission_id=sub.id, **card_from_json(card, source_bench))
        sub.verified = True
        sub.is_published = True
        db.commit()
        prim = next(r for r in sub.results if r.metric_name == bm.primary_metric)
        print(f"  {bm.slug:12s} {name:42s} {bm.primary_metric}={prim.value:.4f} "
              f"[{prim.ci_low:.4f}, {prim.ci_high:.4f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true")
    a = ap.parse_args()
    settings.eval_backend = "sync"
    init_db()
    db = SessionLocal()
    try:
        org = db.query(User).filter(User.email == ORG["email"]).first()
        if org is None:
            org = User(password_hash=hash_password(secrets.token_urlsafe(24)), role="maintainer", **ORG)
            db.add(org)
            db.commit()
        for spec in BENCHMARKS:
            bm = db.query(Benchmark).filter(Benchmark.slug == spec["slug"]).first()
            if bm is None:
                bm = Benchmark(slug=spec["slug"])
                db.add(bm)
            for field, value in spec.items():
                setattr(bm, field, value)
            bm.split_version = "v2"
            bm.starter_url = KAGGLE.format((spec["config"] or {}).get("ground_truth") or spec["slug"])
            bm.is_active = True
            db.commit()
            bm.stats = manifest_stats(bm)
            db.commit()
            print(f"[{bm.slug}] {'ready' if is_ready(bm) else 'NO GROUND TRUTH — listed, not accepting'}")
            if is_ready(bm):
                seed_baselines(db, org, bm, a.rescore)
    finally:
        db.close()


if __name__ == "__main__":
    main()
