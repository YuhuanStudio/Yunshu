"""CPU-only TensorFold round accounting from existing, qualified snapshot receipts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DONE = re.compile(
    r"prompt=(\d+) cached=(\d+).*?tokens=(\d+).*?finish=(\w+).*?rounds=(\d+) accepted=(\d+)/(\d+)"
)


def parse_rounds(log: str, depth: int) -> list[dict]:
    found = []
    for line in log.splitlines():
        m = DONE.search(line)
        if not m:
            continue
        pt, cached, tokens, finish, rounds, accepted, drafted = m.groups()
        if int(pt) != depth:
            continue
        values = [int(x) for x in (pt, cached, tokens, rounds, accepted, drafted)]
        if finish != "length" or values[2] != 2048 or values[3] <= 0:
            raise ValueError("incomplete TensorFold decode")
        found.append(
            dict(
                zip(
                    ("pt", "cached", "tokens", "rounds", "accepted", "drafted"),
                    values,
                    strict=True,
                )
            )
        )
    if len(found) != 2:
        raise ValueError("expected exactly cold and warm round receipts")
    return found


def round_metrics(tokens: int, rounds: int, total_s: float, ttft_s: float) -> dict:
    if tokens < 2 or rounds <= 0 or total_s <= ttft_s:
        raise ValueError("invalid round timing")
    return {
        "outputs_per_round": (tokens - 1) / rounds,
        "wall_ms_per_round": (total_s - ttft_s) * 1000 / rounds,
        "scope": "round wall incl draft/verify/keep/host; NOT isolated target forward",
    }


def accepted_receipt(job: dict) -> bool:
    """Only the scheduler receipt decides contention, never an output's stale flag."""
    return (
        job.get("state") == "done"
        and job.get("rc") == 0
        and job.get("contended") is False
        and job.get("swap_mb", float("inf")) <= 64
    )


def analyze(root: Path, tokenizer, jobs: Path) -> dict:
    cells = []
    state = json.loads((root / "state.json").read_text())
    for depth in (32768, 65536, 131072):
        for rep in range(3):
            rows = {}
            for engine in ("tf-native", "yunshu-tf4", "yunshu-main"):
                paths = sorted(
                    (root / "runs" / engine).glob(
                        f"{engine}-L{depth}-r{rep}-try*.jsonl"
                    )
                )
                qualified = []
                for path in paths:
                    entry = state.get(path.stem)
                    if not entry:
                        raise ValueError("missing scheduler job ID")
                    job = json.loads((jobs / (entry["id"] + ".json")).read_text())
                    if not accepted_receipt(job):
                        continue
                    rs = [json.loads(line) for line in path.read_text().splitlines()]
                    if not any(r.get("complete") is True for r in rs):
                        continue
                    ds = [r for r in rs if r.get("part") == "decode"]
                    if len(ds) != 3 or any(
                        r.get("contended") or r.get("finish") != "length" for r in ds
                    ):
                        continue
                    qualified.append((path, ds, entry["id"]))
                # First qualifying attempt matches bench016 policy, retain path provenance.
                if not qualified:
                    raise ValueError(f"missing complete {engine}/{depth}/{rep}")
                path, ds, job_id = qualified[0]
                rows[engine] = {r["phase"]: r for r in ds}
                tf_stats = None
                if engine == "tf-native":
                    logs = list(
                        (root / "work" / path.stem / "out").glob("server-*.log")
                    )
                    if len(logs) != 1:
                        raise ValueError("missing unique TensorFold server log")
                    tf_stats = parse_rounds(logs[0].read_text(), depth)
                for phase in ("cold", "warm", "turn2"):
                    r = rows[engine][phase]
                    if r["ct"] != (256 if phase == "turn2" else 2048):
                        raise ValueError("short output")
                    s = r.get("xy", {}).get("speculative", {}) if r.get("xy") else {}
                    if tf_stats and phase != "turn2":
                        s = tf_stats[0 if phase == "cold" else 1]
                    cells.append(
                        {
                            "engine": engine,
                            "depth": depth,
                            "rep": rep,
                            "phase": phase,
                            "path": str(path),
                            "job_id": job_id,
                            "ct": r["ct"],
                            "sha": r["sha"],
                            "decode_tps": r["dec_tps"],
                            "ttft_s": r["ttft_s"],
                            "cached": r["cached"],
                            "rounds": s.get("rounds"),
                            "accepted": s.get("accepted"),
                            "drafted": s.get("drafted"),
                            "copy": s.get("copy"),
                            **(
                                round_metrics(
                                    r["ct"], s["rounds"], r["total_s"], r["ttft_s"]
                                )
                                if s.get("rounds")
                                else {}
                            ),
                        }
                    )
            # Compare token IDs in memory; write only digests/counts, never prompt/completion text.
            a = tokenizer.encode(
                rows["tf-native"]["cold"]["text"], add_special_tokens=False
            ).ids
            b = tokenizer.encode(
                rows["yunshu-tf4"]["cold"]["text"], add_special_tokens=False
            ).ids
            if min(len(a), len(b)) < 512:
                raise ValueError("missing first 512 tokens")
            prefix = next(
                (
                    i
                    for i, pair in enumerate(zip(a, b, strict=False))
                    if pair[0] != pair[1]
                ),
                min(len(a), len(b)),
            )
            cells.append(
                {
                    "depth": depth,
                    "rep": rep,
                    "comparison": "TF vs Yunshu TF4 cold",
                    "first512_equal": a[:512] == b[:512],
                    "common_prefix_tokens": prefix,
                    "retokenized": True,
                    "note": "decoded-text re-tokenization; not raw generation IDs",
                }
            )
    return {
        "complete": True,
        "source": str(root),
        "cells": cells,
        "qualification": "derived from bench016 accepted records; original gpuq receipts authoritative",
        "new_gpu_measurement": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--jobs", type=Path, default=Path("/Volumes/P5Plus/yunshu-gpuq/jobs")
    )
    args = p.parse_args()
    from tokenizers import Tokenizer

    result = analyze(args.root, Tokenizer.from_file(str(args.tokenizer)), args.jobs)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
