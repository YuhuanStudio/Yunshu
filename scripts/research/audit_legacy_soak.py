#!/usr/bin/env python3
"""Pinned E04 soak stage in an isolated gate root, on an audit port only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
MODEL = Path("/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--minutes", type=float, default=240)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.minutes <= 0:
        ap.error("positive duration required")
    gate = ROOT / "scripts/release/gate.sh"
    subprocess.run(["zsh", "-n", str(gate)], check=True)
    for path in (
        MODEL / "config.json",
        MAIN / "reference/omlx/omlx/eval/data/mmlu_pro_test.jsonl",
        Path("/Users/yuhuan/Downloads/Qwen3.8-27B-oQ4e-mtp_mmlu_pro.json"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(
            json.dumps(
                {"complete": "dry-run", "minutes": args.minutes, "stage": "soak"}
            )
        )
        return
    if args.smoke:
        import tfbench
        from audit_flag_matrix import TINY

        tfbench.OUT = args.out.parent / "tiny-soak-server"
        tfbench.YUNSHU_SRC = str(ROOT / "python")
        output = args.out.with_suffix(".jsonl")
        server = tfbench.Srv(
            "yunshu", {"YUNSHU_VLM_DRAFT": "mtp"}, "soak-smoke", model=str(TINY)
        )
        try:
            rc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/research/soak_realistic.py"),
                    "--url",
                    server.url,
                    "--model",
                    server.model,
                    "--pid",
                    str(server.proc.pid),
                    "--minutes",
                    "0.1",
                    "--idle-s",
                    "0",
                    "--final-idle-s",
                    "0",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
            ).returncode
        finally:
            server.kill()
        rows = [
            json.loads(line)
            for line in output.read_text().splitlines()
            if line.startswith("{")
        ]
        complete = rc == 0 and bool(rows) and rows[-1].get("kind") == "summary"
        result = {
            "complete": complete,
            "rc": rc,
            "tiny_smoke": True,
            "summary": rows[-1] if rows else None,
        }
    else:
        from audit_flag_matrix import free_port

        base = args.out.parent / "gate"
        binary = base / "root/bin-vision/yunshu"
        binary.parent.mkdir(parents=True, exist_ok=True)
        if not binary.exists():
            binary.symlink_to(MAIN / ".venv/bin/yunshu")
        env = dict(
            os.environ,
            STAGE="soak",
            SOAK_MINUTES=str(args.minutes),
            PORT=str(free_port()),
            OUT=str(base / "out"),
            GATE_ROOT=str(base / "root"),
            GATE_MODELS_DIR="/Volumes/P5Plus/models",
            M=str(MODEL),
            MMLU_IDS="/Users/yuhuan/Downloads/Qwen3.8-27B-oQ4e-mtp_mmlu_pro.json",
            PY=sys.executable,
            PYTHONPATH=str(ROOT / "python"),
        )
        rc = subprocess.run(["zsh", str(gate)], cwd=ROOT, env=env).returncode
        summaries = sorted((base / "out").glob("results-*.jsonl"))
        results = [
            json.loads(line)
            for path in summaries
            for line in path.read_text().splitlines()
            if line.startswith("{")
        ]
        realistic = base / "out/soak-realistic.jsonl"
        rows = (
            [
                json.loads(line)
                for line in realistic.read_text().splitlines()
                if line.startswith("{")
            ]
            if realistic.exists()
            else []
        )
        complete = (
            rc == 0
            and bool(results)
            and bool(rows)
            and rows[-1].get("kind") == "summary"
        )
        result = {
            "complete": complete,
            "rc": rc,
            "stage": "soak",
            "minutes": args.minutes,
            "gate_results": results,
            "realistic_summary": rows[-1] if rows else None,
            "source": str(ROOT / "python"),
        }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
