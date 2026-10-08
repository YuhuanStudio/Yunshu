"""Pinned-tree client compatibility pilot used by yv; CPU-checkable before queueing.

Runs the m3sweep route registry on one small model and proves the source tree SHA.
No timing claims: this is served-route correctness evidence on either device.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

CHECKS = (
    "agent-custom-tools",
    "agent-shell-search",
    "agent-documents-citations",
    "agent-anthropic-client-tools",
    "agent-continuous-usage",
    "agent-template-props",
    "agent-http-video",
)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tree-sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=("m3", "m5"), required=True)
    ap.add_argument("--dry-run", action="store_true")
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
        if v.get("status") not in ("pass", "skip")
    }
    if failed:
        return False, str(failed)
    return True, ""


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
    import m3sweep_jobs as sweep

    if a.device == "m3":
        env_out = str(Path(a.out).with_suffix(".env.json"))
        if sweep.main(["env", "--sync", "--out", env_out]) not in (None, 0):
            raise SystemExit("M3 environment sync failed before server start")

    result = sweep.main(
        ["routes", "--model", a.model, "--only", ",".join(CHECKS), "--out", a.out]
    )
    path = Path(a.out)
    data = json.loads(path.read_text())
    data.update(device=a.device, tree_sha=tree)
    if a.device == "m3":
        data["environment"] = json.loads(Path(env_out).read_text())
    ok, reason = judge(data)
    data["pass"] = bool(ok)
    if not ok:
        data.setdefault("failures", []).append(reason)
    path.write_text(json.dumps(data, indent=2))
    return 0 if ok and result in (None, 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
