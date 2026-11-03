#!/usr/bin/env python3
"""Sampler A/B for the bimodal Yunshu DFlash decode (research, not part of the fair comparison).

Hypothesis under test: the slow runs (DFlash verify ~139 ms/round vs ~44 ms with identical rounds,
drafted and accepted counts) are caused by the telemetry sampler, specifically its `footprint -p`
probe (~69 ms per call), or by ioreg, or by external load.

Each config is one FRESH yunshu-main server (isolated HOME, same launch as the harness via
bench_engines.build_launch) that serves:
  1. one cold code-1024 request (tfbench decode prompt + LONG_ASK, out 128, greedy),
  2. one turn-2 request (same length, ignore_eos).
Configs:
  A  no sampler
  B  sampler, all probes (the harness default)
  C  sampler without the footprint probe (fullcmp_telemetry.Sampler(probes=...))

Records per server: config, rep, index, wall start, top-5 CPU processes right before the launch,
dec tok/s for both requests, every "DFlash round profile" line from the server log, and the sampler
probe set actually started. Any invalid request (short reply, stream error, missing profile line
when a sampler config is requested) raises: the run stops on the first invalid rep, exit 1.
Every server started here is SIGKILLed on exit.

Usage (inside a gpuq job):
    fullcmp_sampler_ab.py --seq AB --tag pairA
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("TFB_YUNSHU_BIN", "/Volumes/P5Plus/yunshu-build/venv-fullcmp-yunshu/bin/yunshu")
os.environ.setdefault("TFB_YUNSHU_SRC", "/Volumes/P5Plus/yunshu-build/wt-fullcmp-yunshu/python")

import bench_engines as be  # noqa: E402
import fullcmp_telemetry as tm  # noqa: E402
import tfbench as tb  # noqa: E402

TREE = "/Volumes/P5Plus/yunshu-build/wt-fullcmp-yunshu"
RAW = Path("/Volumes/Micron/yunshu-fullcmp/sampler_ab")
PORT = 18991  # one port, reused: each server is killed and confirmed closed before the next starts
OUT_TOKENS = 128
FOOTPRINT_FREE = tuple(p for p in tm.ALL_PROBES if p != "footprint")
CONFIGS = {"A": None, "B": tm.ALL_PROBES, "C": FOOTPRINT_FREE}
PROFILE_RE = re.compile(
    r"DFlash round profile: (\d+) rounds, ([\d.]+) nodes/round, draft ([\d.]+) ms, verify ([\d.]+) ms"
)


def top5_cpu() -> list[str]:
    out = subprocess.run(
        ["ps", "-Ao", "pid,pcpu,comm", "-r"], capture_output=True, text=True, timeout=10
    ).stdout.splitlines()
    return [ln.strip() for ln in out[1:6]]


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_ready(proc, url: str, log: Path, timeout: float = 900.0) -> str:
    deadline = time.monotonic() + timeout
    req = urllib.request.Request(url + "/v1/models", headers={"Authorization": "Bearer k"})
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup; see {log}")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode())["data"]
                if data:
                    return data[0]["id"]
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"server not ready after {timeout:.0f}s; see {log}")


def run_one(letter: str, idx: int, tag: str, out_path: Path) -> dict:
    run_dir = RAW / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    home = run_dir / f"home-{idx}-{letter}"
    if home.exists():
        shutil.rmtree(home)
    home.mkdir(parents=True)
    url = f"http://127.0.0.1:{PORT}"
    if port_open(PORT):
        raise RuntimeError(f"port {PORT} already in use before launch; refusing to continue")

    environ = dict(os.environ, TFB_YUNSHU_BIN=os.environ["TFB_YUNSHU_BIN"],
                   TFB_YUNSHU_SRC=os.environ["TFB_YUNSHU_SRC"])
    launch = be.build_launch("yunshu-main", PORT, home, environ)
    for path, text in launch.files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    for link, target in launch.links.items():
        Path(link).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.lexists(link):
            os.symlink(target, link)

    top = top5_cpu()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S")
    log = run_dir / f"server-{idx}-{letter}.log"
    proc = None
    sampler = None
    rec: dict = {"config": letter, "idx": idx, "tag": tag, "wall_start": wall_start, "top5_cpu": top}
    try:
        with open(log, "wb") as fh:
            proc = subprocess.Popen(
                launch.cmd, stdout=fh, stderr=subprocess.STDOUT, env=launch.env, cwd=TREE
            )
        model = wait_ready(proc, url, log)
        rec["model_id"] = model
        if CONFIGS[letter] is not None:
            sampler = tm.Sampler(
                proc.pid, PORT, str(run_dir / f"telemetry-{idx}-{letter}.jsonl"),
                probes=CONFIGS[letter],
            )
            sampler.start()
            rec["probes_started"] = sorted(
                n for n, st in sampler._state.items() if st.available
            )
            if letter == "C" and "footprint" in rec["probes_started"]:
                raise RuntimeError("config C started the footprint probe")

        text = tb.decode_prompt("code", 1024, True)
        cold_body = tb.req(model, text, OUT_TOKENS)
        cold = tb.send(url, cold_body)
        tb.check_decode_len(cold, OUT_TOKENS, f"cold {letter}{idx}")
        turn2_body = tb.req(model, text, OUT_TOKENS)
        turn2_body["ignore_eos"] = True
        turn2_body["messages"] += [
            {"role": "assistant", "content": cold["_text"]},
            {"role": "user", "content": "Continue with the next part, at the same length."},
        ]
        turn2 = tb.send(url, turn2_body)
        tb.check_decode_len(turn2, OUT_TOKENS, f"turn2 {letter}{idx}")
        rec.update(
            cold_dec_tps=cold.get("dec_tps"), cold_ttft_s=cold.get("ttft_s"), cold_ct=cold.get("ct"),
            turn2_dec_tps=turn2.get("dec_tps"), turn2_ttft_s=turn2.get("ttft_s"),
            turn2_ct=turn2.get("ct"),
        )
    finally:
        if sampler is not None:
            try:
                footer = sampler.stop()
                rec["telemetry_footer_ok"] = bool(footer)
            except Exception as exc:  # noqa: BLE001 - recorded, the server kill still runs
                rec["telemetry_error"] = repr(exc)
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=30)
        for _ in range(50):
            if not port_open(PORT):
                break
            time.sleep(0.2)
        rec["server_rc"] = proc.returncode if proc is not None else None

    text_log = log.read_text(errors="replace") if log.exists() else ""
    profiles = [m.group(0) for m in PROFILE_RE.finditer(text_log)]
    rec["profile_lines"] = profiles
    parsed = [PROFILE_RE.search(p).groups() for p in profiles]
    rec["verify_ms_rounds"] = [float(g[3]) for g in parsed]
    rec["draft_ms_rounds"] = [float(g[2]) for g in parsed]
    if letter != "A" and not profiles:
        raise RuntimeError(f"no DFlash round profile line for {letter}{idx}; see {log}")
    shutil.rmtree(home, ignore_errors=True)
    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("config", "idx", "wall_start", "cold_dec_tps",
                                          "turn2_dec_tps", "server_rc")}), flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seq", required=True, help="config letters in order, e.g. AB or CA")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    bad = [c for c in args.seq if c not in CONFIGS]
    if bad:
        print(f"unknown config(s): {bad}", file=sys.stderr)
        return 2
    out_path = RAW / args.tag / "results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, letter in enumerate(args.seq):
        try:
            run_one(letter, idx, args.tag, out_path)
        except Exception as exc:  # noqa: BLE001 - fail fast on the first invalid rep
            print(f"INVALID REP {letter}{idx}: {exc!r}", file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
