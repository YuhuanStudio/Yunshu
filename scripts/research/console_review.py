"""Real-load review of the console (run through gpuq, priority 0, not --quiet).

    gpuq submit --label console-review --timeout 20 -- python scripts/research/console_review.py RUN_NAME

Starts the real 27B engine and the console process, drives a bursty realistic load (1-8 concurrent
requests: short chat, 8K / 32K prompts with prefix-cache hits and misses, long streams, tool calls,
cancels mid-stream, one failing request, idle gaps) and, while it runs, walks every console page in
four viewports (desktop, iPad Pro 11 portrait / landscape, iPhone 15) taking a screenshot every
500 ms and a layout-shift log (frontend/scripts/review-capture.mjs). The engine's /status stream and
finished-request stream are recorded (metadata only) as a replay for the regression spec.

Output: /Volumes/P5Plus/yunshu-build/console-review/RUN_NAME/{<viewport>/<page>/f-*.jpg, shifts.json,
sheets/<viewport>-<page>.jpg, replay.jsonl, plan.json, summary.json}.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = Path("/Volumes/P5Plus/yunshu-build/console-review")
VIEWPORTS = ("desktop", "ipad-portrait", "ipad-landscape", "iphone")
MODEL = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
_FORBIDDEN = ("prompt", "preview", "content", "message", "text", "input", "output")


@dataclass
class Action:
    at: float  # seconds after load start
    kind: str  # short | p8k | p32k | long | tool | cancel | fail
    params: dict = field(default_factory=dict)


def build_plan(seed: int, seconds: float) -> list[Action]:
    """A deterministic bursty plan: bursts of 1-8 concurrent requests separated by idle gaps.

    Every kind appears at least once for runs of 5+ minutes; 'hit' prompts reuse an earlier prompt's
    prefix (prefix-cache hit), 'miss' prompts start from a fresh prefix. Exactly one request fails."""
    rng = random.Random(seed)
    plan: list[Action] = []
    t = 4.0
    burst = 0
    prefixes: dict[str, int] = {}
    mix = ["short", "short", "short", "p8k", "long", "tool", "cancel", "p32k"]
    while t < seconds - 20:
        n = rng.choice([1, 2, 3, 4, 6, 8])
        for i in range(n):
            kind = mix[(burst + i) % len(mix)] if burst < len(mix) else rng.choice(mix)
            params: dict = {}
            if kind in ("p8k", "p32k"):
                fam = f"{kind}-{rng.randint(0, 1)}"
                params["family"] = fam
                params["hit"] = fam in prefixes
                prefixes[fam] = prefixes.get(fam, 0) + 1
                params["tokens"] = 8000 if kind == "p8k" else 32000
            if kind == "long":
                params["max_tokens"] = 600
            if kind == "cancel":
                params["after_events"] = rng.randint(8, 40)
            plan.append(Action(round(t + rng.uniform(0, 1.5), 2), kind, params))
        burst += 1
        t += rng.uniform(25, 55)  # idle gap before the next burst
    if len(plan) > 3:
        plan[len(plan) // 2] = Action(plan[len(plan) // 2].at, "fail", {})
    plan.sort(key=lambda a: a.at)
    return plan


def strip_meta(obj):
    """Drop anything that could hold prompt or reply text from a recorded engine payload."""
    if isinstance(obj, dict):
        return {
            k: strip_meta(v)
            for k, v in obj.items()
            if not any(f in k.lower() for f in _FORBIDDEN)
            or k.lower().endswith("tokens")
        }
    if isinstance(obj, list):
        return [strip_meta(v) for v in obj]
    return obj


def sheet_indices(n_frames: int, max_cells: int = 24) -> list[int]:
    """Evenly spaced frame indices (always including first and last) for a contact sheet."""
    if n_frames <= max_cells:
        return list(range(n_frames))
    return sorted(
        {round(i * (n_frames - 1) / (max_cells - 1)) for i in range(max_cells)}
    )


def make_sheet(frames: list[Path], out: Path, cols: int = 6, cell_w: int = 320) -> bool:
    from PIL import Image  # noqa: PLC0415

    picks = [frames[i] for i in sheet_indices(len(frames))]
    if not picks:
        return False
    imgs = []
    for p in picks:
        im = Image.open(p).convert("RGB")
        imgs.append(im.resize((cell_w, max(1, round(im.height * cell_w / im.width)))))
    rows = -(-len(imgs) // cols)
    h = max(i.height for i in imgs)
    sheet = Image.new("RGB", (cols * cell_w, rows * h), (20, 20, 20))
    for k, im in enumerate(imgs):
        sheet.paste(im, ((k % cols) * cell_w, (k // cols) * h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=70)
    return True


def _filler(tokens: int, family: str) -> str:
    unit = (
        f"[{family}] The quick brown fox jumps over the lazy dog near the river bank. "
    )
    return unit * max(1, tokens // 16)


def _request(base: str, model: str, a: Action, results: list[dict]) -> None:
    host, port = base.split("//")[1].split(":")
    body: dict = {"model": model, "stream": True, "max_tokens": 64}
    msgs = [{"role": "user", "content": "Say hello in one short sentence."}]
    if a.kind in ("p8k", "p32k"):
        msgs = [
            {
                "role": "user",
                "content": _filler(a.params["tokens"], a.params["family"])
                + f"\nSummarise in one sentence. (#{int(a.at)})",
            }
        ]
        body["max_tokens"] = 48
    elif a.kind == "long":
        msgs = [{"role": "user", "content": "Write a detailed essay about tides."}]
        body["max_tokens"] = a.params["max_tokens"]
    elif a.kind == "tool":
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        msgs = [
            {"role": "user", "content": "What is the weather in Taipei? Use the tool."}
        ]
    elif a.kind == "cancel":
        msgs = [{"role": "user", "content": "Count slowly from 1 to 500."}]
        body["max_tokens"] = 500
    elif a.kind == "fail":
        msgs = []  # an empty conversation is refused with a 4xx by every OpenAI-compatible server
    body["messages"] = msgs
    t0 = time.time()
    status, events, ok = 0, 0, False
    try:
        c = http.client.HTTPConnection(host, int(port), timeout=300)
        c.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(body),
            {"content-type": "application/json"},
        )
        r = c.getresponse()
        status = r.status
        for raw in r:
            if raw.startswith(b"data:"):
                events += 1
                if a.kind == "cancel" and events >= a.params["after_events"]:
                    break  # drop the connection mid-stream
        c.close()
        ok = True
    except Exception as exc:  # noqa: BLE001
        results.append({"kind": a.kind, "error": repr(exc)[:120]})
        return
    results.append(
        {
            "kind": a.kind,
            "at": a.at,
            "status": status,
            "events": events,
            "s": round(time.time() - t0, 2),
            "ok": ok,
        }
    )


def _get(url: str, timeout: float = 5.0):
    import urllib.request  # noqa: PLC0415

    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def record_replay(engine: str, path: Path, stop: threading.Event) -> None:
    """Engine /status every 250 ms and finished requests (cursor) every second, metadata only."""
    seq = None
    t0 = time.time()
    last_req = 0.0
    with path.open("w") as f:
        while not stop.is_set():
            now = time.time()
            try:
                f.write(
                    json.dumps(
                        {
                            "t": round(now - t0, 3),
                            "status": strip_meta(_get(f"{engine}/v1/yunshu/status", 2)),
                        }
                    )
                    + "\n"
                )
                if now - last_req >= 1.0:
                    last_req = now
                    q = f"?after_seq={seq}" if seq is not None else "?limit=64"
                    page = _get(f"{engine}/v1/yunshu/requests/recent{q}", 2)
                    seq = page.get("latest_seq", seq)
                    if page.get("data"):
                        f.write(
                            json.dumps(
                                {
                                    "t": round(now - t0, 3),
                                    "requests": strip_meta(page["data"]),
                                }
                            )
                            + "\n"
                        )
                f.flush()
            except Exception:  # noqa: BLE001
                pass
            stop.wait(0.25)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--engine-port", type=int, default=18990)
    ap.add_argument("--console-port", type=int, default=18991)
    ap.add_argument("--seconds", type=float, default=540)
    ap.add_argument("--page-seconds", type=int, default=36)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--viewports", default=",".join(VIEWPORTS))
    ap.add_argument("--startup-timeout", type=float, default=420)
    a = ap.parse_args(argv)
    out = OUT_ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan(a.seed, a.seconds)
    (out / "plan.json").write_text(json.dumps([asdict(p) for p in plan], indent=1))
    home = out / "tmp"
    home.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "python"),
        "YUNSHU_CONSOLE_DB": str(home / "history.sqlite"),
        "YUNSHU_CONSOLE_PORT": str(a.console_port),
    }
    engine = f"http://127.0.0.1:{a.engine_port}"
    console = f"http://127.0.0.1:{a.console_port}"
    procs: list[subprocess.Popen] = []
    stop = threading.Event()
    try:
        build = subprocess.run(
            ["node", "./node_modules/vite/bin/vite.js", "build"],
            cwd=ROOT / "frontend",
            capture_output=True,
            text=True,
        )
        if build.returncode:
            print("FAIL frontend build\n" + build.stderr[-800:])
            return 1
        log = open(home / "engine.log", "wb")  # noqa: SIM115
        eng = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "yunshu_cli",
                "serve",
                "-m",
                a.model,
                "--port",
                str(a.engine_port),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append(eng)
        t0 = time.time()
        while True:
            if eng.poll() is not None:
                print("FAIL engine exited early")
                return 1
            if time.time() - t0 > a.startup_timeout:
                print("FAIL engine did not come up")
                return 1
            try:
                if _get(f"{engine}/health", 3).get("status") and _get(
                    f"{console}/v1/yunshu/console", 3
                ).get("up"):
                    break
            except Exception:  # noqa: BLE001
                time.sleep(1.0)
        model_id = _get(f"{engine}/v1/models")["data"][0]["id"]
        print(f"engine up in {time.time() - t0:.0f}s, model {model_id}", flush=True)
        # warm the model so the recording starts with a ready engine
        r0: list[dict] = []
        _request(engine, model_id, Action(0, "short"), r0)
        if not r0 or not r0[0].get("ok") or r0[0].get("status") != 200:
            print(f"FAIL warm-up request: {r0}")
            return 1

        rec = threading.Thread(
            target=record_replay, args=(engine, out / "replay.jsonl", stop), daemon=True
        )
        rec.start()
        caps = []
        for vp in a.viewports.split(","):
            cap_log = open(out / f"capture-{vp}.log", "wb")  # noqa: SIM115
            caps.append(
                subprocess.Popen(
                    [
                        "node",
                        "scripts/review-capture.mjs",
                        "--base",
                        console,
                        "--out",
                        str(out),
                        "--viewport",
                        vp,
                        "--seconds",
                        str(a.page_seconds),
                    ],
                    cwd=ROOT / "frontend",
                    stdout=cap_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        procs.extend(caps)

        results: list[dict] = []
        threads: list[threading.Thread] = []
        start = time.time()
        for act in plan:
            while time.time() - start < act.at:
                time.sleep(0.1)
            th = threading.Thread(
                target=_request, args=(engine, model_id, act, results), daemon=True
            )
            th.start()
            threads.append(th)
            print(f"[{time.time() - start:5.0f}s] {act.kind} {act.params}", flush=True)
        for th in threads:
            th.join(timeout=300)
        for c in caps:
            c.wait(timeout=1200)
        stop.set()
        rec.join(timeout=5)
        bad = [
            r
            for r in results
            if r["kind"] != "fail" and (not r.get("ok") or r.get("status") != 200)
        ]
        fails = [r for r in results if r["kind"] == "fail"]
        for vp in a.viewports.split(","):
            for pdir in sorted((out / vp).glob("*/")):
                make_sheet(
                    sorted(pdir.glob("f-*.jpg")),
                    out / "sheets" / f"{vp}-{pdir.name}.jpg",
                )
        summary = {
            "requests": len(results),
            "bad": bad,
            "failed_on_purpose": fails,
            "capture_rc": [c.returncode for c in caps],
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
        if (
            bad
            or any(c.returncode for c in caps)
            or not fails
            or fails[0].get("status") in (None, 200)
        ):
            print("FAIL review run")
            return 1
        print(f"PASS review run: {out}")
        return 0
    finally:
        stop.set()
        for p in reversed(procs):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, 15)
        time.sleep(3)
        for p in procs:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, 9)


if __name__ == "__main__":
    raise SystemExit(main())
