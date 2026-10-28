"""Console push vs poll: requests per tab, server CPU per tab, event-to-screen latency, reconnect.

    gpuq submit --label front-pushbench --priority 1 --timeout 20 -- python scripts/research/front_push_bench.py OUTDIR

Starts an engine (`yunshu serve --no-console`) and a STANDALONE console process in front of it (so the
engine can be restarted under a resident console), then for each cell {tabs 1, 10} x {poll (?push=0),
push} runs frontend/scripts/push-bench.mjs (Playwright tabs) while sampling the console and engine
process CPU, then a reconnect cell per mode: the engine is SIGKILLed and restarted under one open tab.
Exits non-zero on the first failed cell. Every process it starts is killed on exit.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from console_real_server import stop_tree  # noqa: E402

MODEL = "/Volumes/P5Plus/models/Qwen2.5-3B-Instruct-4bit"
STATIC = Path("/Volumes/P5Plus/yunshu-build/front-dist-b")


def window_cpu(samples: list[tuple[float, float]], t0: float, t1: float) -> dict:
    """CPU% statistics of (time, cpu%) samples inside [t0, t1] (pure; unit-tested)."""
    v = [c for t, c in samples if t0 <= t <= t1]
    if not v:
        return {"n": 0}
    return {
        "n": len(v),
        "mean_pct": round(statistics.fmean(v), 3),
        "max_pct": round(max(v), 3),
    }


class Sampler(threading.Thread):
    """1 Hz CPU% of one pid (psutil), as (epoch seconds, percent of one core)."""

    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        import psutil

        self.p = psutil.Process(pid)
        self.samples: list[tuple[float, float]] = []
        self.stop = threading.Event()

    def run(self) -> None:
        last = sum(self.p.cpu_times()[:2])
        last_t = time.time()
        while not self.stop.wait(1.0):
            try:
                now = sum(self.p.cpu_times()[:2])
            except Exception:  # noqa: BLE001
                return
            t = time.time()
            self.samples.append((t, 100.0 * (now - last) / max(1e-6, t - last_t)))
            last, last_t = now, t


def get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


class Stack:
    def __init__(self, a: argparse.Namespace, home: Path) -> None:
        self.a = a
        self.home = home
        self.env = {
            **os.environ,
            "HOME": str(home / "home"),
            "PYTHONPATH": str(ROOT / "python"),
            "YUNSHU_CONSOLE_DB": str(home / "history.sqlite"),
            "YUNSHU_DEBUG_ROUTES": "1",
            "YUNSHU_AUTH_DISABLED": "true",  # loopback bench: model load is an admin route
        }
        (home / "home").mkdir(parents=True, exist_ok=True)
        self.engine: subprocess.Popen | None = None
        self.console: subprocess.Popen | None = None
        self.engine_url = f"http://127.0.0.1:{a.engine_port}"
        self.console_url = f"http://127.0.0.1:{a.console_port}"

    def models_dir(self) -> Path:
        """Multi-model mode (a model load needs the manager): one folder of symlinks, ids = names."""
        d = self.home / "models"
        d.mkdir(exist_ok=True)
        for src in (self.a.model, self.a.load_model):
            link = d / Path(src).name
            if not link.exists():
                link.symlink_to(src)
        return d

    def start_engine(self) -> None:
        log = open(self.home / "engine.log", "ab")  # noqa: SIM115
        which = (
            ["--models-dir", str(self.models_dir())]
            if self.a.load_model
            else ["-m", self.a.model]
        )
        self.engine = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "yunshu_cli",
                "serve",
                *which,
                "--port",
                str(self.a.engine_port),
                "--no-console",
            ],
            env=self.env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.wait_engine()

    def wait_engine(self) -> None:
        t0 = time.time()
        while time.time() - t0 < self.a.startup_timeout:
            if self.engine and self.engine.poll() is not None:
                raise SystemExit(
                    f"FAIL engine exited early; see {self.home}/engine.log"
                )
            try:
                if get(f"{self.engine_url}/health", 3).get("status"):
                    return
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        raise SystemExit("FAIL engine did not come up")

    def start_console(self) -> None:
        log = open(self.home / "console.log", "ab")  # noqa: SIM115
        self.console = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "yunshu_console",
                "--engine",
                self.engine_url,
                "--port",
                str(self.a.console_port),
                "--static-dir",
                str(STATIC),
            ],
            env=self.env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        t0 = time.time()
        while time.time() - t0 < 60:
            try:
                if get(f"{self.console_url}/v1/yunshu/console", 3).get("up"):
                    return
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        raise SystemExit("FAIL console did not come up")

    def engine_pid(self) -> int:
        return int(get(f"{self.engine_url}/v1/yunshu/status")["pid"])

    def stop(self) -> None:
        for p in (self.console, self.engine):
            if p is not None:
                stop_tree(p)


def node(a: argparse.Namespace, out: Path, extra: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "node",
            "scripts/push-bench.mjs",
            "--base",
            f"http://127.0.0.1:{a.console_port}",
            "--out",
            str(out),
            *extra,
        ],
        cwd=ROOT / "frontend",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def run_events(
    stack: Stack, a, model_id: str, tabs: int, push: int, outdir: Path
) -> dict:
    out = outdir / f"events-tabs{tabs}-push{push}.json"
    assert stack.console is not None
    cpu_console = Sampler(stack.console.pid)
    cpu_engine = Sampler(stack.engine_pid())
    cpu_console.start()
    cpu_engine.start()
    extra = [
        "--tabs",
        str(tabs),
        "--push",
        str(push),
        "--model",
        model_id,
        "--idle-s",
        str(a.idle_s),
        "--cycles",
        str(a.cycles),
    ]
    if a.load_model:
        extra += ["--load-model", Path(a.load_model).name]
    p = node(a, out, extra)
    try:
        text, _ = p.communicate(timeout=900)
    except subprocess.TimeoutExpired:
        p.kill()
        raise SystemExit(f"FAIL node timeout tabs={tabs} push={push}") from None
    finally:
        cpu_console.stop.set()
        cpu_engine.stop.set()
    if p.returncode != 0 or not out.exists():
        raise SystemExit(
            f"FAIL node rc={p.returncode} tabs={tabs} push={push}\n{text[-1500:]}"
        )
    res = json.loads(out.read_text())
    ev0, ev1 = res["eventsStart"] / 1000, res["eventsEnd"] / 1000
    i0, i1 = res["idleStart"] / 1000, res["idleEnd"] / 1000
    res["cpu"] = {
        "console_idle": window_cpu(cpu_console.samples, i0, i1),
        "console_events": window_cpu(cpu_console.samples, ev0, ev1),
        "engine_idle": window_cpu(cpu_engine.samples, i0, i1),
        "engine_events": window_cpu(cpu_engine.samples, ev0, ev1),
    }
    # fail closed: a push cell must really have used the stream, a poll cell must not
    live = [s for s in res["pushState"] if s]
    flat = [x for s in live for x in s]
    if push and not any(x.get("events", 0) > 0 for x in flat):
        raise SystemExit(
            f"FAIL push cell tabs={tabs} received no stream events: {flat}"
        )
    if not push and flat:
        raise SystemExit(f"FAIL poll cell opened a stream: {flat}")
    if push and not res["latencyMs"]["requestArrival_sent_to_pill"]:
        raise SystemExit(f"FAIL no arrival latency measured tabs={tabs} push={push}")
    if a.load_model and (res.get("modelLoad") or {}).get("http") != 200:
        raise SystemExit(f"FAIL model load did not succeed: {res.get('modelLoad')}")
    out.write_text(json.dumps(res, indent=1))
    return res


def run_reconnect(stack: Stack, a, push: int, outdir: Path) -> dict:
    out = outdir / f"reconnect-push{push}.json"
    p = node(
        a,
        out,
        ["--mode", "reconnect", "--tabs", "1", "--push", str(push), "--seconds", "70"],
    )
    assert p.stdout is not None
    line = p.stdout.readline()
    if "READY" not in line:
        p.kill()
        raise SystemExit(f"FAIL reconnect node not ready: {line}")
    time.sleep(10)
    pid = stack.engine_pid()
    os.kill(pid, signal.SIGKILL)
    killed = time.time()
    with contextlib.suppress(Exception):
        stack.engine.wait(timeout=5)  # type: ignore[union-attr]
    time.sleep(3)
    stack.start_engine()
    restarted = time.time()
    rest = p.communicate(timeout=200)[0]
    if p.returncode != 0 or not out.exists():
        raise SystemExit(f"FAIL reconnect node rc={p.returncode}\n{rest[-1500:]}")
    res = json.loads(out.read_text())
    res["driver"] = {"killed": killed, "engine_ready": restarted}
    out.write_text(json.dumps(res, indent=1))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--load-model", default="")
    ap.add_argument("--engine-port", type=int, default=18990)
    ap.add_argument("--console-port", type=int, default=18991)
    ap.add_argument("--idle-s", type=int, default=40)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--startup-timeout", type=float, default=300)
    ap.add_argument("--tabs", default="1,10")
    ap.add_argument("--skip-reconnect", action="store_true")
    a = ap.parse_args(argv)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="front-bench-", dir=str(outdir)))
    stack = Stack(a, home)
    summary: dict = {}
    try:
        stack.start_engine()
        stack.start_console()
        ids = [m["id"] for m in get(f"{stack.engine_url}/v1/models")["data"]]
        model_id = Path(a.model).name if a.load_model else ids[0]
        if model_id not in ids:
            raise SystemExit(f"FAIL model {model_id} not in {ids}")
        # warm the model once so the first cell does not pay for it
        urllib.request.urlopen(
            urllib.request.Request(
                f"{stack.engine_url}/v1/chat/completions",
                json.dumps(
                    {
                        "model": model_id,
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                ).encode(),
                {"content-type": "application/json"},
            ),
            timeout=120,
        ).read()
        for tabs in [int(x) for x in a.tabs.split(",")]:
            for push in (0, 1):
                r = run_events(stack, a, model_id, tabs, push, outdir)
                summary[f"tabs{tabs}-push{push}"] = {
                    k: r.get(k)
                    for k in ("idle", "events", "latencyMs", "modelLoad", "cpu")
                }
                print(f"cell tabs={tabs} push={push} done", flush=True)
        if not a.skip_reconnect:
            for push in (0, 1):
                r = run_reconnect(stack, a, push, outdir)
                summary[f"reconnect-push{push}"] = r["reconnect"] | {
                    "counts": r.get("reconnectCounts")
                }
                print(f"reconnect push={push} done", flush=True)
        (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
        return 0
    finally:
        stack.stop()


if __name__ == "__main__":
    raise SystemExit(main())
