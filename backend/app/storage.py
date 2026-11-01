"""Storage abstraction — filesystem now, S3 later without touching callers."""
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from app.config import settings


class Storage(ABC):
    @abstractmethod
    def save(self, key: str, stream: BinaryIO, max_bytes: int | None = None) -> tuple[str | None, str, int]:
        """Persist stream under key. Returns (uri, sha256, size); uri is None if
        max_bytes was exceeded (nothing is kept)."""

    @abstractmethod
    def path(self, uri: str) -> Path:
        """Local path for a stored object (evaluators read files directly)."""

    @abstractmethod
    def delete(self, uri: str) -> None:
        """Remove a stored object if it exists."""


class FilesystemStorage(Storage):
    def __init__(self, root: str):
        self.root = Path(root)

    def save(self, key: str, stream: BinaryIO, max_bytes: int | None = None) -> tuple[str | None, str, int]:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        sha, size = hashlib.sha256(), 0
        with open(dest, "wb") as f:
            while chunk := stream.read(1 << 20):  # 1 MB chunks — never buffer the whole file
                size += len(chunk)
                if max_bytes is not None and size > max_bytes:
                    break
                sha.update(chunk)
                f.write(chunk)
        if max_bytes is not None and size > max_bytes:
            dest.unlink(missing_ok=True)
            return None, "", size
        return f"file://{dest.as_posix()}", sha.hexdigest(), size

    def path(self, uri: str) -> Path:
        return Path(uri.removeprefix("file://"))

    def delete(self, uri: str) -> None:
        self.path(uri).unlink(missing_ok=True)


storage: Storage = FilesystemStorage(settings.storage_dir)
