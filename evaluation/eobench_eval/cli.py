"""`eobench` command-line client (stdlib only).

    eobench validate predictions.csv --benchmark eurosat [--hub URL | --manifest m.csv --classes a,b,…]
    eobench submit   predictions.csv --benchmark eurosat --name "My model"   (needs EOB_TOKEN)
    eobench status   <submission-id>

`validate` runs the server's structural checks locally — header, columns,
probabilities, label vocabulary and exact coverage of the hidden ids — using
only public information (the public split manifest and the class list), so a
malformed file never reaches the server.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

DEFAULT_HUB = os.environ.get("EOB_HUB", "http://localhost:8000")


def _get(url: str, token: str | None = None) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def _public_ids(manifest: Path) -> set[str]:
    with open(manifest, newline="") as f:
        return {r["sample_id"] for r in csv.DictReader(f) if r["split"] == "test_hidden"}


def cmd_validate(a) -> int:
    from eobench_eval import get_evaluator

    if a.manifest and a.classes:
        task, classes, config = a.task, a.classes.split(","), {}
        manifest = Path(a.manifest)
    else:
        bm = _get(f"{a.hub}/api/benchmarks/{a.benchmark}")
        task, classes, config = bm["task_type"], bm["classes"], bm.get("config") or {}
        src = config.get("ground_truth") or a.benchmark
        tmp = Path(tempfile.mkdtemp()) / "manifest.csv"
        urllib.request.urlretrieve(f"{a.hub}/media/{src}/split_manifest_{bm['split_version']}.csv", tmp)
        manifest = tmp
    ev = get_evaluator(task)
    ids = sorted(_public_ids(manifest))
    cfg = {**config, "classes": classes}
    gt = {"ids": ids, "zip": None} if task == "semantic_segmentation" else {i: classes[0] for i in ids}
    if task == "semantic_segmentation":
        # without hidden masks only names can be checked locally
        import zipfile
        names = {Path(n).stem for n in zipfile.ZipFile(a.file).namelist() if n.lower().endswith(".png")}
        errors = []
        if set(ids) - names:
            errors.append(f"{len(set(ids) - names)} missing tile(s)")
        if names - set(ids):
            errors.append(f"{len(names - set(ids))} unknown tile(s)")
    else:
        errors = ev.validate(Path(a.file), gt, cfg)
    if errors:
        print("INVALID:\n  " + "\n  ".join(errors))
        return 1
    print(f"OK: {a.file} is a valid {task} submission covering all {len(ids)} hidden ids")
    return 0


def cmd_submit(a) -> int:
    token = os.environ.get("EOB_TOKEN")
    if not token:
        sys.exit("Set EOB_TOKEN to an API token from your hub account page.")
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in (("model_name", a.name), ("description", a.description)):
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    p = Path(a.file)
    body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{p.name}"\r\n'
               f"Content-Type: application/octet-stream\r\n\r\n".encode())
    body.write(p.read_bytes())
    body.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"{a.hub}/api/benchmarks/{a.benchmark}/submissions", data=body.getvalue(),
                                 method="POST", headers={"Authorization": f"Bearer {token}",
                                                         "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req) as r:
            sub = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"error {e.code}: {json.load(e).get('detail')}")
    print(f"submitted {sub['id']} — {a.hub}/submissions/{sub['id']}")
    a.id = sub["id"]
    return cmd_status(a, wait=True)


def cmd_status(a, wait: bool = False) -> int:
    token = os.environ.get("EOB_TOKEN")
    while True:
        sub = _get(f"{a.hub}/api/submissions/{a.id}", token)
        if not wait or sub["status"] in ("COMPLETED", "REJECTED", "FAILED"):
            break
        time.sleep(1.5)
    print(f"status: {sub['status']}")
    for e in (sub.get("error_report") or {}).get("errors") or []:
        print("  error:", e)
    for r in sub["results"]:
        ci = f"  [{r['ci_low']:.4f}, {r['ci_high']:.4f}]" if r.get("ci_low") is not None else ""
        print(f"  {r['metric_name']:<26} {r['value']:.4f}{ci}")
    return 0 if sub["status"] == "COMPLETED" else 1


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--hub", default=DEFAULT_HUB, help="hub URL (or $EOB_HUB)")
    ap = argparse.ArgumentParser(prog="eobench", description="EO-Bench Hub client")
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate", help="check a prediction file locally", parents=[common])
    v.add_argument("file")
    v.add_argument("--benchmark", default="eurosat")
    v.add_argument("--manifest", help="public split manifest (offline mode)")
    v.add_argument("--classes", help="comma-separated class list (offline mode)")
    v.add_argument("--task", default="scene_classification")
    s = sp.add_parser("submit", help="upload predictions and wait for the score", parents=[common])
    s.add_argument("file")
    s.add_argument("--benchmark", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--description", default="")
    st = sp.add_parser("status", help="show a submission's status and metrics", parents=[common])
    st.add_argument("id")
    a = ap.parse_args()
    sys.exit({"validate": cmd_validate, "submit": cmd_submit, "status": cmd_status}[a.cmd](a))


if __name__ == "__main__":
    main()
