"""Evaluator plugin contract.

Every task type implements this ABC. The same code runs in three places:
the evaluation worker, local validation in a researcher's notebook, and the
golden-test suite.
"""
from abc import ABC, abstractmethod
from pathlib import Path


class SubmissionError(ValueError):
    """The uploaded file is malformed. Raised from evaluate() for problems that
    only surface while reading the full file; the worker turns it into a
    REJECTED status (the user's fault) rather than FAILED (ours)."""


class Evaluator(ABC):
    task_type: str
    version: str
    #: name -> {label, description, higher_is_better, unit: "ratio"|"value"}
    metrics: dict[str, dict] = {}
    #: file extensions accepted for upload
    formats: tuple[str, ...] = ("csv",)

    def required_files(self, config: dict) -> list[str]:
        """Ground-truth files (relative to the benchmark's gt dir) that must
        exist before the benchmark can accept submissions."""
        return [f"split_manifest_{config.get('split_version', 'v2')}.csv"]

    def format_spec(self, config: dict) -> dict:
        """Machine-readable submission format, rendered on the benchmark page."""
        return {}

    @abstractmethod
    def load_ground_truth(self, gt_dir: Path, config: dict) -> object:
        """Load ground truth for a benchmark from its server-side directory."""

    @abstractmethod
    def validate(self, pred_path: Path, gt: object, config: dict) -> list[str]:
        """Cheap structural checks. Empty list = valid."""

    @abstractmethod
    def evaluate(self, pred_path: Path, gt: object, config: dict) -> dict:
        """Compute metrics. Returns {"metrics": {...}, "ci": {...}, "extra": {...}}."""


def read_manifest_rows(gt_dir: Path, config: dict, split: str = "test_hidden") -> list[dict]:
    import csv

    manifest = gt_dir / f"split_manifest_{config.get('split_version', 'v2')}.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Ground truth manifest missing: {manifest}")
    with open(manifest, newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]
