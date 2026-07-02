"""Storage abstraction — filesystem now, S3 later without touching callers."""
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from app.config import settings


class Storage(ABC):
    @abstractmethod
    def save(self, key: str, stream: BinaryIO) -> tuple[str, str]:
        """Persist stream under key. Returns (uri, sha256)."""

    @abstractmethod
    def path(self, uri: str) -> Path:
        """Local path for a stored object (workers read files directly)."""


class FilesystemStorage(Storage):
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, stream: BinaryIO) -> tuple[str, str]:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256()
        with open(dest, "wb") as f:
            while chunk := stream.read(1 << 20):  # 1 MB chunks — never buffer whole file
                sha.update(chunk)
                f.write(chunk)
        return f"file://{dest}", sha.hexdigest()

    def path(self, uri: str) -> Path:
        return Path(uri.removeprefix("file://"))


storage: Storage = FilesystemStorage(settings.storage_dir)
