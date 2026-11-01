"""Shared helpers for the Kaggle training kernels."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("EOB_OUT", "/kaggle/working/out"))
SMOKE = os.environ.get("EOB_SMOKE") == "1"  # tiny local CPU run to catch bugs before GPU time
T0 = time.time()


def log(*args) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m]", *args, flush=True)


def elapsed_h() -> float:
    return (time.time() - T0) / 3600


def find_input(owner_slug: str) -> Path:
    """Locate an attached Kaggle dataset (mount layout changed over the years)."""
    owner, slug = owner_slug.split("/")
    env = os.environ.get(f"EOB_DATA_{slug.upper().replace('-', '_').replace('.', '_')}")
    candidates = [Path(env)] if env else []
    candidates += [Path("/kaggle/input/datasets") / owner / slug, Path("/kaggle/input") / slug]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Dataset {owner_slug} not found in {candidates}")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def pip_install(*pkgs: str) -> bool:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log("pip install failed:", pkgs, r.stderr[-500:])
    return r.returncode == 0


def environment() -> dict:
    import torch

    info = {"torch": torch.__version__, "python": sys.version.split()[0]}
    try:
        import timm

        info["timm"] = timm.__version__
    except ImportError:
        pass
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return info
