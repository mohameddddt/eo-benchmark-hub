"""Evaluator plugin contract.

Every task type implements this ABC. The same code runs in three places:
the Celery worker, `eobench validate` in Colab, and the golden-test suite.
"""
from abc import ABC, abstractmethod
from pathlib import Path


class Evaluator(ABC):
    task_type: str
    version: str

    @abstractmethod
    def load_ground_truth(self, gt_dir: Path, config: dict) -> object:
        """Load ground truth for a benchmark from its server-side directory."""

    @abstractmethod
    def validate(self, pred_path: Path, gt: object, config: dict) -> list[str]:
        """Cheap structural checks. Empty list = valid."""

    @abstractmethod
    def evaluate(self, pred_path: Path, gt: object, config: dict) -> dict:
        """Compute metrics. Returns {"metrics": {...}, "extra": {...}}."""
