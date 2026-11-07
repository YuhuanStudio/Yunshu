#!/usr/bin/env python3
"""YUNSHU_OBSERVE on/off A/B for the pinned Yunshu decode (research, not part of the fair comparison).

Question: does observe mode (YUNSHU_OBSERVE=1) cost speed on a watched engine? tfbench.py starts every
watched yunshu engine with YUNSHU_OBSERVE=1 unless TFB_OBSERVE=0, so the timed reps run observe-on by
default. This driver measures what that costs.

Arms (letters as used by --seq):
  A  off: YUNSHU_OBSERVE unset in the server environment
  B  on:  YUNSHU_OBSERVE=1 in the server environment

Each rep is one FRESH pinned yunshu-main server (isolated HOME, same launch as the harness via
bench_engines.build_launch, same port and kill discipline as fullcmp_sampler_ab.py). Per server:
  1. one cold request: tfbench decode prompt (code-1024 or code-8192) + LONG_ASK, out 256, greedy;
  2. one turn-2 request: the same prompt, the cold reply as assistant turn, a continue ask, ignore_eos.
Recorded per server: dec tok/s, TTFT for both requests, every DFlash round profile line (rounds,
nodes/round, draft ms, verify ms per round), the quiet-machine check taken right before the launch,
and the /debug/engine status code (observe-on must answer 200, observe-off 404: proof the arm engaged).

Invalid rep (short reply, no round profile line, engagement mismatch, server exit, port busy): raise,
exit 1 on the first one. Every server started here is SIGKILLed on exit. Quiet-machine foreign holders
do NOT invalidate a rep; they are recorded and flagged (`quiet_flag`).

Outliers are decided by `summarize`, not during the run: a server whose mean verify ms/round is more
than 1.5x the median of all servers for that prompt is counted and excluded from the means.

Usage (inside a gpuq job; --seq must be an AB pair so the global order is ABAB...):
    fullcmp_observe_ab.py run --prompt code-1024 --seq AB --idx0 0 --tag c1024-p0
    fullcmp_observe_ab.py summarize
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_engines as be  # noqa: E402
import fullcmp_quiet as fq  # noqa: E402
import fullcmp_sampler_ab as ab  # noqa: E402  (launch pins, port helpers, profile regex)
import tfbench as tb  # noqa: E402

PORT = 18993  # 18990-18998 are this experiment's range; one port, each server killed before the next
OUT_TOKENS = 256
ARMS = {"A": "off", "B": "on"}
PROMPTS = {"code-1024": ("code", 1024), "code-8192": ("code", 8192)}
OUTLIER_FACTOR = 1.5
RAW = Path("/Volumes/Micron/yunshu-fullcmp/observe_ab")
COMPSYNC = Path("/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103")
THRESHOLDS_FILE = COMPSYNC / "rep_thresholds.json"
OUT_JSON = COMPSYNC / "observe_ab.json"
OBSERVE_ADDS = [
    "YUNSHU_DEBUG_ROUTES=1: mounts the /debug/* routes (engine, system, kv-cache, spec-decode, ...)",
    "YUNSHU_TELEMETRY=on: 1 Hz unprivileged Apple power/GPU/temperature sampler (settings.py: +1.8% follow-up TTFT at 1K on the 27B A/B)",
    "YUNSHU_SERVE_LOG=1: one numbers-only JSON line per finished request to a rotated file under HOME/.yunshu/logs",
    "YUNSHU_AUDIT_LOG_FILE=HOME/.yunshu/logs/audit.log: audit log file written",
    "YUNSHU_REPLAY_TOKEN_IDS=1: exact token ids in logprobs",
    "YUNSHU_DEBUG_STREAM_CAPTURE=HOME/.yunshu/logs/stream_capture.jsonl: generation capture file (VLM-runner generations per settings.py)",
]


def arm_env(launch_env: dict, arm: str) -> dict:
    """The server environment for one arm. build_launch scrubs YUNSHU_* from the parent
    environment, so the arm's value is set here, after the scrub, and never inherited."""
    env = {k: v for k, v in launch_env.items() if k != "YUNSHU_OBSERVE"}
    if ARMS[arm] == "on":
        env["YUNSHU_OBSERVE"] = "1"
    return env


def debug_status(url: str) -> int:
    req = urllib.request.Request(url + "/debug/engine", headers={"Authorization": "Bearer k"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code


def expected_debug_status(arm: str) -> int:
    return 200 if ARMS[arm] == "on" else 404


def parse_profiles(log_text: str) -> list[dict]:
    out = []
    for m in ab.PROFILE_RE.finditer(log_text):
        rounds, nodes, draft, verify = m.groups()
        out.append({"rounds": int(rounds), "nodes_per_round": float(nodes),
                    "draft_ms": float(draft), "verify_ms": float(verify)})
    return out


def quiet_record() -> dict:
    rec = fq.check()  # 10 s window, read-only; never raises on a busy host
    return {
        "ok": bool(rec.get("ok")),
        "total_foreign_pct": rec.get("total_foreign_pct"),
        "holders": [{"name": h.get("name"), "pct": h.get("pct")} for h in rec.get("holders", [])],
    }


def run_one(prompt: str, letter: str, idx: int, tag: str) -> dict:
    arm = letter  # the arm letter (A off, B on) selects the environment
    kind, ctx = PROMPTS[prompt]
    run_dir = RAW / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    home = run_dir / f"home-{idx}-{letter}"
    if home.exists():
        shutil.rmtree(home)
    home.mkdir(parents=True)
    url = f"http://127.0.0.1:{PORT}"
    if ab.port_open(PORT):
        raise RuntimeError(f"port {PORT} already in use before launch; refusing to continue")

    quiet = quiet_record()
    environ = dict(
        os.environ,
        TFB_YUNSHU_BIN=os.environ.get("TFB_YUNSHU_BIN", ab.PIN_YUNSHU_BIN),
        TFB_YUNSHU_SRC=os.environ.get("TFB_YUNSHU_SRC", ab.PIN_YUNSHU_SRC),
    )
    launch = be.build_launch("yunshu-main", PORT, home, environ)
    launch.env = arm_env(launch.env, arm)
    for path, text in launch.files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    for link, target in launch.links.items():
        Path(link).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.lexists(link):
            os.symlink(target, link)

    top = ab.top5_cpu()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S")
    log = run_dir / f"server-{idx}-{letter}.log"
    rec: dict = {"prompt": prompt, "arm": ARMS[arm], "letter": letter, "idx": idx, "tag": tag,
                 "wall_start": wall_start, "top5_cpu": top, "quiet": quiet,
                 "quiet_flag": not quiet["ok"], "out_tokens": OUT_TOKENS}
    proc = None
    try:
        with open(log, "wb") as fh:
            proc = subprocess.Popen(
                launch.cmd, stdout=fh, stderr=subprocess.STDOUT, env=launch.env, cwd=ab.TREE
            )
        rec["model_id"] = ab.wait_ready(proc, url, log)
        status = debug_status(url)
        rec["debug_engine_status"] = status
        if status != expected_debug_status(arm):
            raise RuntimeError(
                f"{letter}{idx} arm {ARMS[arm]}: /debug/engine answered {status}, "
                f"expected {expected_debug_status(arm)} (arm did not engage as set)"
            )

        text = tb.decode_prompt(kind, ctx, True)
        cold = tb.send(url, tb.req(rec["model_id"], text, OUT_TOKENS))
        tb.check_decode_len(cold, OUT_TOKENS, f"cold {letter}{idx} {prompt}")
        turn2_body = tb.req(rec["model_id"], text, OUT_TOKENS)
        turn2_body["ignore_eos"] = True
        turn2_body["messages"] += [
            {"role": "assistant", "content": cold["_text"]},
            {"role": "user", "content": "Continue with the next part, at the same length."},
        ]
        turn2 = tb.send(url, turn2_body)
        tb.check_decode_len(turn2, OUT_TOKENS, f"turn2 {letter}{idx} {prompt}")
        rec.update(
            cold_dec_tps=cold.get("dec_tps"), cold_ttft_s=cold.get("ttft_s"), cold_ct=cold.get("ct"),
            turn2_dec_tps=turn2.get("dec_tps"), turn2_ttft_s=turn2.get("ttft_s"),
            turn2_ct=turn2.get("ct"),
        )
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=30)
        for _ in range(50):
            if not ab.port_open(PORT):
                break
            time.sleep(0.2)
        rec["server_rc"] = proc.returncode if proc is not None else None

    text_log = log.read_text(errors="replace") if log.exists() else ""
    profiles = parse_profiles(text_log)
    rec["profiles"] = profiles
    if not profiles:
        raise RuntimeError(f"no DFlash round profile line for {letter}{idx} {prompt}; see {log}")
    rec["verify_ms_round"] = round(statistics.fmean(p["verify_ms"] for p in profiles), 3)
    rec["draft_ms_round"] = round(statistics.fmean(p["draft_ms"] for p in profiles), 3)
    shutil.rmtree(home, ignore_errors=True)
    out_path = RAW / tag / "results.jsonl"
    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("prompt", "arm", "idx", "cold_dec_tps", "turn2_dec_tps",
                                          "verify_ms_round", "quiet_flag")}), flush=True)
    return rec


def run_cmd(prompt: str, seq: str, idx0: int, tag: str) -> int:
    if prompt not in PROMPTS:
        print(f"unknown prompt {prompt!r}", file=sys.stderr)
        return 2
    if not seq or any(c not in ARMS for c in seq) or len(seq) % 2 or seq[0] != "A":
        print("seq must be an even-length AB pair starting with A (global order ABAB...)",
              file=sys.stderr)
        return 2
    for i, letter in enumerate(seq):
        try:
            run_one(prompt, letter, idx0 + i, tag)
        except Exception as exc:  # noqa: BLE001 - fail fast on the first invalid rep
            print(f"INVALID REP {letter}{idx0 + i} {prompt}: {exc!r}", file=sys.stderr, flush=True)
            return 1
    return 0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def load_records() -> list[dict]:
    recs = []
    for path in sorted(RAW.glob("*/results.jsonl")):
        if path.parent.name.startswith("pilot"):
            continue  # the one-rep pilot is a check, not data
        for line in path.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(statistics.fmean(values), 3),
            "median": round(statistics.median(values), 3),
            "min": round(min(values), 3), "max": round(max(values), 3)}


def _pct_gap(off: float, on: float, *, higher_is_better: bool) -> float:
    """Observe-on cost in percent; positive = observe slower."""
    if higher_is_better:
        return round((off - on) / off * 100, 3)
    return round((on - off) / off * 100, 3)


def summarize_prompt(prompt: str, recs: list[dict], thr: dict) -> dict:
    rows = [r for r in recs if r["prompt"] == prompt]
    med = statistics.median(r["verify_ms_round"] for r in rows)
    for r in rows:
        r["outlier"] = r["verify_ms_round"] > OUTLIER_FACTOR * med
    kept = [r for r in rows if not r["outlier"]]
    per_arm = {}
    for arm in ("off", "on"):
        a = [r for r in kept if r["arm"] == arm]
        per_arm[arm] = {
            "servers": len([r for r in rows if r["arm"] == arm]),
            "included": len(a),
            "outliers": len([r for r in rows if r["arm"] == arm and r["outlier"]]),
            "quiet_flags": len([r for r in rows if r["arm"] == arm and r["quiet_flag"]]),
            "decode_cold_tps": _stats([r["cold_dec_tps"] for r in a]),
            "decode_turn2_tps": _stats([r["turn2_dec_tps"] for r in a]),
            "ttft_cold_s": _stats([r["cold_ttft_s"] for r in a]),
            "ttft_turn2_s": _stats([r["turn2_ttft_s"] for r in a]),
            "verify_ms_round": _stats([r["verify_ms_round"] for r in a]),
        }
    off, on = per_arm["off"], per_arm["on"]
    overhead = {}
    checks = [
        ("decode_tps", "decode_cold_tps", True, "decode_tps"),
        ("turn2_decode_tps", "decode_turn2_tps", True, "turn2_decode_tps"),
        ("cold_ttft", "ttft_cold_s", False, "cold_ttft"),
        ("turn2_ttft", "ttft_turn2_s", False, "turn2_ttft"),
    ]
    within = True
    for name, key, hib, thr_key in checks:
        if off[key].get("n", 0) == 0 or on[key].get("n", 0) == 0:
            overhead[name] = {"pct": None, "threshold_pct": thr[thr_key], "within": False}
            within = False
            continue
        pct = _pct_gap(off[key]["mean"], on[key]["mean"], higher_is_better=hib)
        ok = abs(pct) <= thr[thr_key]
        within = within and ok
        overhead[name] = {"pct": pct, "threshold_pct": thr[thr_key], "within": ok}
    return {"per_arm": per_arm, "overhead": overhead, "within_noise": within,
            "verify_median_all_servers_ms": round(med, 3),
            "outlier_servers": [{"idx": r["idx"], "arm": r["arm"], "tag": r["tag"],
                                 "verify_ms_round": r["verify_ms_round"]}
                                for r in rows if r["outlier"]]}


def summarize_cmd() -> int:
    recs = load_records()
    raw = json.loads(THRESHOLDS_FILE.read_text())["thresholds_pct"]
    thr = {k: raw[k] for k in ("decode_tps", "turn2_decode_tps", "cold_ttft", "turn2_ttft")}
    prompts = {}
    for p in PROMPTS:
        if any(r["prompt"] == p for r in recs):
            prompts[p] = summarize_prompt(p, recs, thr)
    complete = bool(prompts) and all(
        prompts[p]["per_arm"]["off"]["included"] >= 3 and prompts[p]["per_arm"]["on"]["included"] >= 3
        for p in prompts
    ) and set(prompts) == set(PROMPTS)
    if not complete:
        verdict = "insufficient: fewer than 3 included servers per arm for a prompt"
    elif all(v["within_noise"] for v in prompts.values()):
        verdict = "observe-on overhead is inside the noise thresholds for every prompt"
    else:
        bad = [p for p, v in prompts.items() if not v["within_noise"]]
        verdict = f"observe-on overhead exceeds the noise thresholds for {', '.join(bad)}"
    doc = {
        "experiment": "YUNSHU_OBSERVE=1 (B, on) vs unset (A, off), pinned yunshu-main, DFlash decode",
        "observe_on_in_timed_reps": True,
        "observe_on_in_timed_reps_evidence": (
            "scripts/research/tfbench.py: a watched yunshu engine gets YUNSHU_OBSERVE=1 unless "
            "TFB_OBSERVE=0; no script in scripts/ sets TFB_OBSERVE=0"
        ),
        "what_observe_adds": OBSERVE_ADDS,
        "design": {"prompts": {p: {"kind": k, "ctx": c} for p, (k, c) in PROMPTS.items()},
                   "out_tokens": OUT_TOKENS, "order": "ABAB per prompt, fresh server per rep",
                   "outlier_rule": f"mean verify ms/round > {OUTLIER_FACTOR} x median of all servers"},
        "thresholds_pct": thr,
        "prompts": prompts,
        "verdict": verdict,
        "total_servers": len(recs),
        "outliers_total": sum(len(v["outlier_servers"]) for v in prompts.values()),
        "quiet_flags_total": sum(1 for r in recs if r["quiet_flag"]),
        "raw_dir": str(RAW),
    }
    COMPSYNC.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"verdict": verdict, "servers": len(recs), "out": str(OUT_JSON)}))
    return 0 if complete else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--prompt", required=True, choices=sorted(PROMPTS))
    r.add_argument("--seq", required=True, help="AB pair, e.g. AB")
    r.add_argument("--idx0", type=int, required=True)
    r.add_argument("--tag", required=True)
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "run":
        return run_cmd(args.prompt, args.seq, args.idx0, args.tag)
    return summarize_cmd()


if __name__ == "__main__":
    sys.exit(main())
