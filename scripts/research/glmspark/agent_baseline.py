"""Pinned offline category-C baseline; run only via gpuq (tiny first)."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUILD = Path("/Volumes/P5Plus/yunshu-build/codex/glmspark")
PYTHON = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python"
BENCH = BUILD / "tool-eval-bench"
sys.path.insert(0, str(ROOT / "scripts/research/agentic"))
import servers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ, PYTHONPATH=str(BENCH / "src") + os.pathsep + str(ROOT / "python")
    )
    base = [
        PYTHON,
        "-m",
        "tool_eval_bench",
        "--categories",
        "C",
        "--no-live",
        "--no-think",
        "--temperature",
        "0",
        "--no-warmup",
        "--timeout",
        "180",
        "--json-file",
        str(out / "teb.json"),
        "--output-dir",
        str(out / "reports"),
    ]
    if a.tiny:
        base += [
            "--scenarios",
            "TC-09",
            "--max-turns",
            "2",
            "--no-preflight",
            "--backend-kwargs",
            '{"max_tokens":256}',
        ]
    else:
        base += ["--backend-kwargs", '{"max_tokens":4096}']
    subprocess.run(
        [PYTHON, "-c", "import tool_eval_bench, httpx, yaml, rich"], env=env, check=True
    )
    if a.dry_run:
        subprocess.run(base + ["--dry-run"], env=env, check=True)
        return
    servers.SERVER_HOME = out / "server-home"
    os.environ["AGENTIC_YUNSHU_SRC"] = str(ROOT / "python")
    server = servers.Server(
        "yunshu", a.model, servers.free_ports(1)[0], out / "server.log"
    )
    rc = 1
    try:
        server.start()
        cmd = base + ["--base-url", server.url + "/v1", "--model", server.model_id]
        (out / "command.json").write_text(json.dumps(cmd, indent=2))
        with (out / "bench.log").open("w") as log:
            rc = subprocess.run(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        if rc or not (out / "teb.json").exists():
            raise RuntimeError(
                f"benchmark rc={rc}, report={(out / 'teb.json').exists()}"
            )
        result = json.loads((out / "teb.json").read_text())
        log_text = (out / "server.log").read_text(errors="replace")
        modes = [
            line
            for line in log_text.splitlines()
            if "Speculative decoding" in line
            or "drafter" in line.lower()
            or "MTP" in line
        ]
        receipt = {
            "complete": True,
            "rc": rc,
            "model": a.model,
            "model_id": server.model_id,
            "bench_revision": subprocess.check_output(
                ["git", "-C", str(BENCH), "rev-parse", "HEAD"], text=True
            ).strip(),
            "worktree_revision": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "tiny": a.tiny,
            "thinking": False,
            "max_tokens": 256 if a.tiny else 4096,
            "mode_evidence": modes,
            "report": str(out / "teb.json"),
            "report_keys": list(result),
        }
        (out / "complete.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
    finally:
        server.kill()


if __name__ == "__main__":
    main()
