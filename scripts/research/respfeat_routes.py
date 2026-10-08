"""One pinned tiny server, bounded served-route checks, fail at the first defect."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CHECKS = (
    "respfeat-computer",
    "respfeat-citations",
    "respfeat-voices",
    "respfeat-webrtc",
)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tree-sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=("m5",), required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra-pythonpath", default="")
    return ap


def judge(data):
    if data.get("complete") is not True or data.get("pass") is not True:
        return False, str(data.get("failures") or "incomplete route checks")
    rows = data.get("checks", {})
    checked = {key.split("@")[0] for key in rows}
    if set(CHECKS) - checked:
        return False, f"Missing checks: {sorted(set(CHECKS) - checked)}"
    failed = {
        k: v.get("detail")
        for k, v in rows.items()
        if v.get("status") != "pass" or v.get("served") is False
    }
    return (False, str(failed)) if failed else (True, "")


def run_checks(ctx, srv, registry):
    """CPU-testable: no later check executes after the first failure/skip."""
    rows, failures = {}, []
    for name in CHECKS:
        chk = registry[name]
        try:
            if srv.proc.poll() is not None:
                raise RuntimeError("Server died before check")
            chk.fn(ctx)
            if getattr(ctx, "downgraded", False):
                raise RuntimeError("Check did not exercise the served path")
            rows[name] = {"status": "pass", "served": True, "routes": list(chk.routes)}
        except Exception as exc:
            rows[name] = {
                "status": "fail",
                "served": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
            failures.append(f"{name}: {rows[name]['detail']}")
            break
    return rows, failures


def wait_for_port(factory, budget=120, now=time.monotonic, sleep=time.sleep):
    """Wait for the shared worker pool; never retry model/startup failures."""
    deadline = now() + budget
    while True:
        try:
            return factory()
        except RuntimeError as exc:
            if "no free port in 18990-18996" not in str(exc) or now() >= deadline:
                raise
            sleep(min(2, max(0, deadline - now())))


def main(argv=None):
    a = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    tree = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    if tree != a.tree_sha:
        raise SystemExit(f"Source tree mismatch: expected {a.tree_sha}, got {tree}")
    if a.dry_run:
        print(json.dumps({"complete": True, "tree_sha": tree, "checks": list(CHECKS)}))
        return 0
    if Path(a.model).name != "Qwen3.5-0.8B-MLX-bf16":
        raise SystemExit("respfeat is limited to the authorized Qwen3.5-0.8B pilot")
    if a.extra_pythonpath:
        sys.path.insert(0, a.extra_pythonpath)
        os.environ["PYTHONPATH"] = (
            a.extra_pythonpath + os.pathsep + os.environ.get("PYTHONPATH", "")
        )
    import m3sweep_jobs as sweep
    import route_checks as rc
    from covaudit_session import Srv

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "kind": "routes",
        "device": a.device,
        "tree_sha": tree,
        "models": [a.model],
        "checks": {},
        "failures": [],
        "pass": False,
        "complete": False,
    }
    srv, ctx = None, None
    try:
        srv = wait_for_port(
            lambda: Srv(
                a.model,
                str(root / "python"),
                out.parent / "home",
                out.parent / "server.log",
                ["YUNSHU_VLM_APC_DISK=0"],
            )
        )
        srv.wait_ready()
        ctx = sweep.routes_make_ctx(srv, "", "vlm")
        data["checks"], data["failures"] = run_checks(ctx, srv, rc.REGISTRY)
        data["pass"] = not data["failures"] and len(data["checks"]) == len(CHECKS)
        data["notes"] = ctx.notes
    except Exception as exc:
        data["failures"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if ctx:
            ctx.http.close()
            ctx.oa.close()
            ctx.an.close()
        if srv:
            data["server_log_tail"] = srv.log_tail(40)
            srv.kill()
        data["complete"] = True
        out.write_text(json.dumps(data, indent=2))
    ok, reason = judge(data)
    if not ok:
        print(reason, file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
