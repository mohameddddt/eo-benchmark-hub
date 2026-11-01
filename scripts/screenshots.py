#!/usr/bin/env python3
"""Capture website screenshots (README + visual QA).

    python scripts/screenshots.py --base http://127.0.0.1:8000 --out docs/screenshots

Needs `pip install playwright && playwright install chromium` and a running hub.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


def first_submission(base: str, slug: str) -> str | None:
    with urllib.request.urlopen(f"{base}/api/benchmarks/{slug}/leaderboard") as r:
        board = json.load(r)
    return board[0]["submission_id"] if board else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--mobile", action="store_true", help="also capture 390px-wide versions")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sub = first_submission(a.base, "eurosat")
    seg = first_submission(a.base, "landcoverai")
    pages = {
        "home": "/",
        "benchmarks": "/benchmarks",
        "eurosat-leaderboard": "/benchmarks/eurosat",
        "eurosat-analysis": "/benchmarks/eurosat?tab=analysis",
        "eurosat-dataset": "/benchmarks/eurosat?tab=dataset",
        "eurosatc-analysis": "/benchmarks/eurosat-c?tab=analysis",
        "eurosatc-dataset": "/benchmarks/eurosat-c?tab=dataset",
        "resisc45-analysis": "/benchmarks/resisc45?tab=analysis",
        "landcoverai-analysis": "/benchmarks/landcoverai?tab=analysis",
        "submit": "/benchmarks/eurosat?tab=submit",
        "models": "/models",
        "docs": "/docs",
    }
    if sub:
        pages["submission"] = f"/submissions/{sub}"
    if seg:
        pages["submission-seg"] = f"/submissions/{seg}"
    if a.only:
        pages = {k: v for k, v in pages.items() if k in a.only}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        configs = [("light", 1440, False), ("dark", 1440, False)] + ([("light", 390, True)] if a.mobile else [])
        for theme, width, mobile in configs:
            ctx = browser.new_context(viewport={"width": width, "height": 900}, device_scale_factor=1 if not mobile else 2,
                                      color_scheme=theme)
            page = ctx.new_page()
            errors: list[str] = []
            page.on("console", lambda m, errors=errors: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            for name, path in pages.items():
                page.goto(a.base + path, wait_until="networkidle")
                page.wait_for_timeout(700)
                suffix = f"-{theme}" + ("-mobile" if mobile else "")
                page.screenshot(path=str(a.out / f"{name}{suffix}.png"), full_page=True)
                overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1")
                print(f"{name}{suffix}: ok{'  !! horizontal overflow' if overflow else ''}")
            if errors:
                print("console errors:", *sorted(set(errors))[:10], sep="\n  ")
            ctx.close()
        browser.close()


if __name__ == "__main__":
    main()
