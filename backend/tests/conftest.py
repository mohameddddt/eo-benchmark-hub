"""Test harness: an isolated data dir (SQLite file, storage, ground truth) per
session and synchronous evaluation, so the full upload -> score -> leaderboard
path runs without Redis, Postgres or threads."""
import os
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="eobench-test-"))
os.environ.update({
    "DATABASE_URL": f"sqlite:///{(TMP / 'test.db').as_posix()}",
    "EVAL_BACKEND": "sync",
    "STORAGE_DIR": str(TMP / "storage"),
    "GROUND_TRUTH_DIR": str(TMP / "gt"),
    "FRONTEND_DIST": str(TMP / "dist"),
    "WEEKLY_QUOTA": "3",
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Benchmark  # noqa: E402

CLASSES = ["Forest", "River", "Highway"]
HIDDEN = {f"es-{i:012x}": CLASSES[i % 3] for i in range(30)}


@pytest.fixture()
def client():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def benchmark(client):
    """A 30-sample scene-classification benchmark with ground truth installed."""
    gt = TMP / "gt" / "toy"
    gt.mkdir(parents=True, exist_ok=True)
    lines = ["sample_id,relpath,label,split"] + [f"{s},{c}/{s}.jpg,{c},test_hidden" for s, c in HIDDEN.items()]
    (gt / "split_manifest_v2.csv").write_text("\n".join(lines) + "\n")
    db = SessionLocal()
    db.add(Benchmark(slug="toy", name="Toy", task_type="scene_classification", classes=CLASSES,
                     primary_metric="accuracy", split_version="v2"))
    db.commit()
    db.close()
    return "toy"


def signup(client, email="ada@example.com", name="Ada") -> dict:
    client.post("/api/auth/register", json={"email": email, "password": "correct-horse", "name": name})
    tok = client.post("/api/auth/login", json={"email": email, "password": "correct-horse"}).json()
    return {"Authorization": f"Bearer {tok['access_token']}"}


def predictions(n_wrong: int) -> bytes:
    rows = ["sample_id,predicted_label"]
    for i, (s, c) in enumerate(sorted(HIDDEN.items())):
        rows.append(f"{s},{CLASSES[(CLASSES.index(c) + 1) % 3] if i < n_wrong else c}")
    return ("\n".join(rows) + "\n").encode()


def submit(client, headers, slug, content: bytes, name="m", filename="p.csv"):
    return client.post(f"/api/benchmarks/{slug}/submissions", headers=headers,
                       files={"file": (filename, content, "text/csv")}, data={"model_name": name})
