#!/usr/bin/env python3
"""Registers the EuroSAT benchmark and checks its ground truth is in place."""
from pathlib import Path

from app.config import settings
from app.database import SessionLocal, init_db
from app.models import Benchmark

EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                   "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

DESCRIPTION = (
    "Sentinel-2 land use / land cover scene classification: 27,000 RGB patches "
    "(64x64), 10 classes. Evaluated on the hidden 10% of split_v1. "
    "Submit a CSV with columns sample_id,predicted_label covering every "
    "test_hidden id from the published split manifest."
)


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        bm = db.query(Benchmark).filter(Benchmark.slug == "eurosat").first()
        if bm is None:
            bm = Benchmark(slug="eurosat")
            db.add(bm)
        bm.name = "EuroSAT"
        bm.task_type = "scene_classification"
        bm.description = DESCRIPTION
        bm.dataset_url = "https://zenodo.org/records/7711810"
        bm.license = "MIT (EuroSAT RGB)"
        bm.primary_metric = "accuracy"
        bm.classes = EUROSAT_CLASSES
        bm.split_version = "v1"
        bm.evaluator_version = "scene_classification/0.1.0"
        bm.is_active = True
        db.commit()
        print("Benchmark 'eurosat' registered.")

        gt = Path(settings.ground_truth_dir) / "eurosat" / "split_manifest_v1.csv"
        if gt.exists():
            print(f"Ground truth OK: {gt}")
        else:
            print(f"WARNING: ground truth missing — copy split_manifest_v1.csv to {gt}")
            print("         (generate it with scripts/make_splits.py, or copy the one")
            print("          your Colab notebook wrote to Drive)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
