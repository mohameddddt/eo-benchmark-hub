#!/usr/bin/env python3
"""Snapshot every public API response into frontend/public/data/ so the site
can be deployed as a static, read-only showcase (GitHub Pages, Netlify, …).

    python scripts/export_static.py
    cd frontend && VITE_STATIC=1 npm run build

Each GET /api/<path> becomes data/<path>.json; the frontend's static mode
reads those files instead of calling the server.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "evaluation"), str(ROOT)]

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

OUT = ROOT / "frontend" / "public" / "data"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    written = 0
    with TestClient(app) as c:
        def grab(path: str):
            nonlocal written
            r = c.get(f"/api/{path}")
            r.raise_for_status()
            dest = OUT / f"{path}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(r.json(), separators=(",", ":")))
            written += 1
            return r.json()

        grab("stats")
        users, subs = set(), set()
        for bm in grab("benchmarks"):
            grab(f"benchmarks/{bm['slug']}")
            for e in grab(f"benchmarks/{bm['slug']}/leaderboard"):
                subs.add(e["submission_id"])
                users.add(e["user"]["id"])
        for sid in sorted(subs):
            grab(f"submissions/{sid}")
        for uid in sorted(users):
            grab(f"users/{uid}")
    size = sum(f.stat().st_size for f in OUT.rglob("*.json")) / 1024
    print(f"wrote {written} files ({size:.0f} KB) to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
