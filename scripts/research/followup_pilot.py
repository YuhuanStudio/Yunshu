"""27B follow-up TTFT pilot: one cold 32K turn, then one follow-up turn that resends the reply.

Reports the follow-up's cached tokens and the server's `turn stages` line (lookup / restore /
suffix prefill / first token). Exits nonzero when the follow-up is not a stream with a first
token, or when no `turn stages` line was logged for it (measurement fails closed).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORTS = range(18990, 19000)
STAGE_RE = re.compile(
    r"turn stages: ids=(\d+) hit=(\d+) \| standing=([\d.]+) vocab=([\d.]+) add=([\d.]+) "
    r"\(lookup=([\d.]+) plan=([\d.]+) boundaries=([\d.]+) hash=([\d.]+) window=([\d.]+)\) "
    r"wait_first=([\d.]+) emit=([\d.]+) other=([\d.]+) total=([\d.]+) ms"
)
FOLLOW_TAIL = "Continue with the next part, at the same length."


def build_followup(prompt: str, reply: str) -> list[dict]:
    """Turn-2 messages: the cold prompt, the reply as sent back, then the next instruction."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": FOLLOW_TAIL},
    ]


def parse_turn_stages(log_text: str) -> list[dict]:
    """Every `turn stages` line in the server log, as numbers (ms except ids and hit)."""
    keys = (
        "ids",
        "hit",
        "standing",
        "vocab",
        "add",
        "lookup",
        "plan",
        "boundaries",
        "hash",
        "window",
        "wait_first",
        "emit",
        "other",
        "total",
    )
    rows = []
    for m in STAGE_RE.finditer(log_text):
        vals = [float(v) for v in m.groups()]
        rows.append(
            {
                k: (int(v) if k in ("ids", "hit") else v)
                for k, v in zip(keys, vals, strict=True)
            }
        )
    return rows


def cached_tokens(usage: dict | None) -> int:
    """Cached prompt tokens from the OpenAI usage shape (prompt_tokens_details.cached_tokens)."""
    det = (usage or {}).get("prompt_tokens_details") or {}
    return int(det.get("cached_tokens") or 0)


def stream_turn(url: str, body: dict, timeout: float = 900.0) -> dict:
    """POST a streamed chat request; TTFT is the first non-empty content delta."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    parts: list[str] = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and first is None:
                    first = time.perf_counter() - t0
                parts.append(piece)
    if first is None:
        raise RuntimeError("stream had no content delta (no first token)")
    return {
        "ttft_s": first,
        "total_s": time.perf_counter() - t0,
        "text": "".join(parts),
        "usage": usage,
    }


def free_port() -> int:
    for p in PORTS:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port in 18990-18999")


def wait_health(port: int, proc: subprocess.Popen, limit: float = 600.0) -> None:
    end = time.time() + limit
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError("server did not become healthy")


# Runs the CLI with every long-prompt APC lookup under cProfile; the stats go to argv[1].
LOOKUP_PROFILER = """
import cProfile, io, pstats, runpy, sys
from yunshu_engine import apc_manager as _apc
_orig = _apc._Coordinator.lookup
_path = sys.argv.pop(1)
def _profiled(self, token_ids, **kw):
    if len(token_ids) < 16384:
        return _orig(self, token_ids, **kw)
    prof = cProfile.Profile()
    prof.enable()
    try:
        return _orig(self, token_ids, **kw)
    finally:
        prof.disable()
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumtime").print_stats(40)
        with open(_path, "a") as f:
            f.write("== lookup prompt=%d\\n" % len(token_ids) + buf.getvalue())
_apc._Coordinator.lookup = _profiled
sys.argv = ["yunshu_cli"] + sys.argv[1:]
runpy.run_module("yunshu_cli", run_name="__main__")
"""

# Times the round driver's methods on the admit-to-first-token span (the hit_stages window) and
# appends one dict per request to argv[1]; nested calls overlap (add includes _restore).
ROUND_PROFILER = """
import runpy, sys, time
from yunshu_engine import hit_stages as _hs
from yunshu_engine.round_driver import driver as _d
_path = sys.argv.pop(1)
_acc = {}
def _wrap(name):
    orig = getattr(_d.RoundDriver, name)
    def timed(self, *a, **k):
        if not _hs.active():
            return orig(self, *a, **k)
        t = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            _acc[name] = _acc.get(name, 0.0) + (time.perf_counter() - t)
            _acc[name + "#n"] = _acc.get(name + "#n", 0) + 1
    setattr(_d.RoundDriver, name, timed)
for _n in ("add", "_restore", "_prefill_step", "_decode_step", "_decode_step_lone",
           "_draft", "_draw", "_finish_join", "_step_impl", "_logits"):
    _wrap(_n)
_fin = _hs.finish
def _finish(ids, hit, emit_s):
    _fin(ids, hit, emit_s)
    row = {k: (round(v * 1000, 1) if not k.endswith("#n") else v) for k, v in _acc.items()}
    row.update(ids=ids, hit=hit)
    with open(_path, "a") as f:
        f.write(repr(row) + "\\n")
    _acc.clear()
_hs.finish = _finish
sys.argv = ["yunshu_cli"] + sys.argv[1:]
runpy.run_module("yunshu_cli", run_name="__main__")
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    ap.add_argument(
        "--prompt", default="/Volumes/P5Plus/yunshu-build/tfnew/prompts/code-32768.txt"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--think-s",
        type=float,
        default=2.0,
        help="seconds between the reply and the follow-up (the user's reading time; 5 = fullcmp idle5)",
    )
    ap.add_argument(
        "--debug-log",
        action="store_true",
        help="YUNSHU_LOG_LEVEL=DEBUG (APC lookup lines)",
    )
    ap.add_argument(
        "--profile-lookup",
        action="store_true",
        help="cProfile each long APC lookup into out/lookup_profile.txt",
    )
    ap.add_argument(
        "--profile-round",
        action="store_true",
        help="time the round driver's methods on each admit-to-first-token span into out/round_profile.txt",
    )
    ap.add_argument(
        "--retry",
        action="store_true",
        help="send the same follow-up again after the think time (a retry); its output must match",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wt = Path(__file__).resolve().parents[2]
    port = free_port()
    env = dict(os.environ, PYTHONPATH=str(wt / "python"), HOME=str(out / "home"))
    if args.debug_log:
        env["YUNSHU_LOG_LEVEL"] = "DEBUG"
    log = out / "server.log"
    serve = ["serve", "-m", args.model, "--port", str(port)]
    if args.profile_lookup:
        cmd = [
            sys.executable,
            "-c",
            LOOKUP_PROFILER,
            str(out / "lookup_profile.txt"),
            *serve,
        ]
    elif args.profile_round:
        cmd = [
            sys.executable,
            "-c",
            ROUND_PROFILER,
            str(out / "round_profile.txt"),
            *serve,
        ]
    else:
        cmd = [sys.executable, "-m", "yunshu_cli", *serve]
    proc = subprocess.Popen(
        cmd,
        cwd=str(wt),
        env=env,
        stdout=open(log, "w"),  # noqa: SIM115 (the server keeps writing to it after this call)
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    result: dict = {
        "port": port,
        "model": args.model,
        "think_s": args.think_s,
        "debug_log": args.debug_log,
    }
    try:
        wait_health(port, proc)
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        prompt = Path(args.prompt).read_text()
        base = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        cold = stream_turn(
            url, dict(base, messages=[{"role": "user", "content": prompt}])
        )
        result["cold"] = {k: cold[k] for k in ("ttft_s", "total_s", "usage")}
        reply = cold["text"]
        result["reply_chars"] = len(reply)
        result["reply_head"] = reply[:120]
        result["reply_tail"] = reply[
            -120:
        ]  # the trailing newline question (a re-rendered reply)
        time.sleep(
            args.think_s
        )  # the cold reply's store finishes; the user reads the reply
        fu = stream_turn(url, dict(base, messages=build_followup(prompt, reply)))
        result["followup"] = {k: fu[k] for k in ("ttft_s", "total_s", "usage")}
        result["followup_cached_tokens"] = cached_tokens(fu["usage"])
        result["followup_text_tail"] = fu["text"][-120:]
        result["tf_reference_cached_tokens"] = 33283
        if args.retry:
            time.sleep(args.think_s)  # the user asks again after reading
            again = stream_turn(url, dict(base, messages=build_followup(prompt, reply)))
            result["retry"] = {
                "ttft_s": again["ttft_s"],
                "cached": cached_tokens(again["usage"]),
                "same_text": again["text"] == fu["text"],
            }
    finally:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    stages = parse_turn_stages(log.read_text(errors="replace"))
    result["turn_stages"] = stages
    (out / "pilot.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    if not stages:
        print("FAIL: no turn stages line logged", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "followup_ttft_s": result["followup"]["ttft_s"],
                "cached": result["followup_cached_tokens"],
                "stages_last": stages[-1],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
