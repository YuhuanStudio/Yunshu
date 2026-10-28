#!/usr/bin/env python3
"""Does gpuq's job_swap (footprint --swapped) measure real disk swap or compressed/GPU pages?

Starts one `yunshu serve`, samples its process tree once per second (footprint per category, vm_stat
compressor/swap counters, sysctl vm.swapusage) while a short streamed decode runs, and writes
samples.jsonl + summary.json (final record has complete=true). No parsing of the model output.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

FOOT_ROW = re.compile(
    r"^\s*(\d+) B\s+(\d+) B\s+(\d+) B\s+(\d+) B\s+(\d+)\s+(.+?)\s*$", re.MULTILINE
)
VM_KEYS = {
    "Pages free": "free",
    "Pages inactive": "inactive",
    "Pages wired down": "wired",
    "File-backed pages": "file_backed",
    "Anonymous pages": "anonymous",
    "Pages stored in compressor": "compressor_stored",
    "Pages occupied by compressor": "compressor_occupied",
    "Pageouts": "pageouts",
    "Swapins": "swapins",
    "Swapouts": "swapouts",
}


def parse_footprint(text: str) -> dict:
    """Category rows plus TOTAL of `footprint --swapped -f bytes`; raises on a missing TOTAL."""
    cats = {}
    for m in FOOT_ROW.finditer(text):
        dirty, swapped, clean, reclaim, regions, name = m.groups()
        cats[name] = dict(
            dirty=int(dirty),
            swapped=int(swapped),
            clean=int(clean),
            reclaimable=int(reclaim),
            regions=int(regions),
        )
    if "TOTAL" not in cats:
        raise ValueError("no TOTAL row")
    total = cats.pop("TOTAL")
    return dict(total=total, categories=cats)


def parse_vm_stat(text: str) -> dict:
    page = int(re.search(r"page size of (\d+) bytes", text).group(1))
    out = {"page": page}
    for key, name in VM_KEYS.items():
        m = re.search(r"^" + re.escape(key) + r":\s+(\d+)\.", text, re.MULTILINE)
        if m is None:
            raise ValueError(f"vm_stat lacks {key}")
        out[name] = int(m.group(1))
    return out


def parse_swapusage(text: str) -> dict:
    """`vm.swapusage: total = 4096.00M  used = 2953.69M  free = 1142.31M` in MB."""
    m = re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M\s+free = ([\d.]+)M", text)
    if m is None:
        raise ValueError("unparsable vm.swapusage")
    return dict(total_mb=float(m[1]), used_mb=float(m[2]), free_mb=float(m[3]))


def top_swapped(foot: dict, n: int = 4) -> list[tuple[str, int]]:
    rows = sorted(
        ((k, v["swapped"]) for k, v in foot["categories"].items() if v["swapped"]),
        key=lambda kv: -kv[1],
    )
    return rows[:n]


def summarize(samples: list[dict], tokens: list[float]) -> dict:
    """Peak/final swapped footprint vs the host counters, and decode rate per swapped bucket."""
    if not samples:
        raise ValueError("no samples")
    first, last = samples[0], samples[-1]
    peak = max(samples, key=lambda s: s["foot_swapped_bytes"])
    page = first["vm"]["page"]
    out = dict(
        samples=len(samples),
        peak_foot_swapped_mb=peak["foot_swapped_bytes"] / 1e6,
        peak_foot_swapped_at=peak["t"],
        peak_top_categories=peak["top_swapped"],
        final_foot_swapped_mb=last["foot_swapped_bytes"] / 1e6,
        swapouts_delta_mb=(last["vm"]["swapouts"] - first["vm"]["swapouts"])
        * page
        / 1e6,
        swapins_delta_mb=(last["vm"]["swapins"] - first["vm"]["swapins"]) * page / 1e6,
        pageouts_delta_mb=(last["vm"]["pageouts"] - first["vm"]["pageouts"])
        * page
        / 1e6,
        swapusage_used_max_delta_mb=max(s["swapusage"]["used_mb"] for s in samples)
        - first["swapusage"]["used_mb"],
        swapusage_total_mb_max=max(s["swapusage"]["total_mb"] for s in samples),
        compressor_stored_max_mb=max(s["vm"]["compressor_stored"] for s in samples)
        * page
        / 1e6,
        compressor_stored_growth_mb=(
            max(s["vm"]["compressor_stored"] for s in samples)
            - first["vm"]["compressor_stored"]
        )
        * page
        / 1e6,
    )
    # Decode rate in seconds where the process' swapped bytes grew vs did not.
    per_sec: dict[int, int] = {}
    for t in tokens:
        per_sec[int(t)] = per_sec.get(int(t), 0) + 1
    grow, flat = [], []
    for prev, cur in zip(samples, samples[1:], strict=False):
        n = per_sec.get(int(cur["t"]))
        if n is None:
            continue
        (
            grow
            if cur["foot_swapped_bytes"] - prev["foot_swapped_bytes"] > 64e6
            else flat
        ).append(n)
    out["decode_tok_s_when_swapped_growing"] = sum(grow) / len(grow) if grow else None
    out["decode_tok_s_when_swapped_flat"] = sum(flat) / len(flat) if flat else None
    out["decode_seconds_growing"] = len(grow)
    out["decode_seconds_flat"] = len(flat)
    return out


def _run(cmd: list[str]) -> str:
    return subprocess.run(
        cmd, capture_output=True, text=True, check=True, timeout=20
    ).stdout


def children(pid: int) -> list[int]:
    rows = [
        tuple(map(int, line.split()))
        for line in _run(["ps", "-axo", "pid=,ppid="]).splitlines()
    ]
    tree = {pid}
    while True:
        grown = tree | {c for c, p in rows if p in tree}
        if grown == tree:
            return sorted(tree)
        tree = grown


def sample_once(pid: int, t0: float) -> dict:
    foot_total = dict(dirty=0, swapped=0, clean=0, reclaimable=0, regions=0)
    cats: dict[str, dict] = {}
    for p in children(pid):
        try:
            f = parse_footprint(
                _run(["footprint", "--swapped", "-f", "bytes", "-p", str(p)])
            )
        except (subprocess.SubprocessError, ValueError):
            continue
        for k, v in f["total"].items():
            foot_total[k] += v
        for name, row in f["categories"].items():
            acc = cats.setdefault(name, dict.fromkeys(row, 0))
            for k, v in row.items():
                acc[k] += v
    foot = dict(total=foot_total, categories=cats)
    return dict(
        t=round(time.time() - t0, 3),
        foot_swapped_bytes=foot_total["swapped"],
        foot_dirty_bytes=foot_total["dirty"],
        foot_clean_bytes=foot_total["clean"],
        top_swapped=top_swapped(foot),
        vm=parse_vm_stat(_run(["vm_stat"])),
        swapusage=parse_swapusage(_run(["sysctl", "vm.swapusage"])),
    )


def available_bytes() -> int:
    """free + inactive pages, as gpuq's memory-pressure gate counts them."""
    vm = parse_vm_stat(_run(["vm_stat"]))
    return (vm["free"] + vm["inactive"]) * vm["page"]


def hog(target_avail_gb: float, hold_s: float, chunk_mb: int = 256) -> int:
    """Dirty incompressible anonymous memory until free+inactive falls to the target, then hold."""
    block = os.urandom(64 << 20)
    held: list[bytearray] = []
    target = int(target_avail_gb * 1e9)
    deadline = time.time() + hold_s
    while available_bytes() > target and time.time() < deadline:
        chunk = bytearray(chunk_mb << 20)
        for off in range(0, len(chunk), len(block)):
            n = min(len(block), len(chunk) - off)
            chunk[off : off + n] = block[:n]
        held.append(chunk)
    print(f"hog holds {len(held) * chunk_mb} MB", flush=True)
    while time.time() < deadline:
        time.sleep(1)
    return len(held) * chunk_mb


def request_rates(marks: dict[str, float], tokens: list[float], n: int) -> list[dict]:
    """Decode tok/s of each request from the stream-chunk times between its start and end marks."""
    rows = []
    for i in range(n):
        a, b = marks.get(f"request{i}_start"), marks.get(f"request{i}_end")
        if a is None or b is None:
            continue
        inside = [t for t in tokens if a <= t <= b]
        # first chunk is prefill-bound: rate counts from it to the last chunk
        span = inside[-1] - inside[0] if len(inside) > 1 else 0.0
        rows.append(
            dict(
                request=i,
                chunks=len(inside),
                decode_tok_s=(len(inside) - 1) / span if span > 0 else None,
                under_hog=marks.get(f"request{i}_hog", False),
            )
        )
    return rows


def stream_decode(
    url: str, model: str, prompt: str, max_tokens: int, t0: float, sink: list[float]
):
    body = json.dumps(
        dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            temperature=0,
        )
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                sink.append(time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18997)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--requests", type=int, default=2)
    ap.add_argument("--idle-s", type=float, default=20.0)
    ap.add_argument("--hog-avail-gb", type=float, default=0.0)
    ap.add_argument(
        "--hog-after", type=int, default=1, help="requests run before the hog starts"
    )
    ap.add_argument("--hog-hold-s", type=float, default=240.0)
    ap.add_argument("--hog-worker", action="store_true")
    ap.add_argument("--yunshu-bin", default=shutil.which("yunshu") or "yunshu")
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    if a.hog_worker:
        hog(a.hog_avail_gb, a.hog_hold_s)
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    t0 = time.time()
    log = open(out / "server.log", "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        [a.yunshu_bin, "serve", "-m", a.model, "--port", str(a.port)],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    samples: list[dict] = []
    tokens: list[float] = []
    marks: dict[str, float] = {}
    stop = threading.Event()
    sink = (out / "samples.jsonl").open("w")

    def sampler():
        while not stop.is_set():
            started = time.time()
            try:
                s = sample_once(proc.pid, t0)
                samples.append(s)
                sink.write(json.dumps(s) + "\n")
                sink.flush()
            except Exception as e:  # noqa: BLE001 - keep sampling
                sink.write(json.dumps(dict(error=repr(e))) + "\n")
            stop.wait(max(0.0, 1.0 - (time.time() - started)))

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    rc = 1
    try:
        url = f"http://127.0.0.1:{a.port}"
        model = None
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                raise RuntimeError("server exited early")
            try:
                with urllib.request.urlopen(url + "/v1/models", timeout=3) as r:
                    model = json.load(r)["data"][0]["id"]
                    break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        if model is None:
            raise RuntimeError("server not ready in 900 s")
        marks["ready"] = time.time() - t0
        prompt = "Write a long story about a lighthouse keeper. " * 20
        hog_proc = None
        for i in range(a.requests):
            if a.hog_avail_gb and i == a.hog_after and hog_proc is None:
                hog_proc = subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--hog-worker",
                        "--model",
                        "-",
                        "--out",
                        "-",
                    ]
                    + [
                        "--hog-avail-gb",
                        str(a.hog_avail_gb),
                        "--hog-hold-s",
                        str(a.hog_hold_s),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                marks["hog_start"] = time.time() - t0
                while time.time() - t0 < marks["hog_start"] + 120:
                    if available_bytes() <= a.hog_avail_gb * 1e9 * 1.05:
                        break
                    time.sleep(1)
                marks["hog_ready"] = time.time() - t0
            marks[f"request{i}_hog"] = hog_proc is not None
            marks[f"request{i}_start"] = time.time() - t0
            stream_decode(url, model, prompt + f" Variant {i}.", a.tokens, t0, tokens)
            marks[f"request{i}_end"] = time.time() - t0
        time.sleep(a.idle_s)
        if hog_proc is not None:
            hog_proc.terminate()
            hog_proc.wait(timeout=60)
        rc = 0
    except Exception as e:  # noqa: BLE001
        marks["error"] = repr(e)
    finally:
        stop.set()
        th.join(timeout=10)
        sink.close()
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    try:
        summary = summarize(samples, tokens)
    except ValueError as e:
        summary = dict(error=str(e))
    summary.update(
        model=a.model,
        marks=marks,
        requests=request_rates(marks, tokens, a.requests),
        tokens=len(tokens),
        complete=rc == 0,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
