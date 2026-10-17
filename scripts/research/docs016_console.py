"""Capture the public console images under real load. Run only through gpuq."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from scripts.research.agentic.servers import free_ports
from scripts.research.console_real_server import get, post, stop_tree

ROOT = Path(__file__).resolve().parents[2]


def validate(expected: dict, actual: dict, requests: int) -> list[str]:
    errors = []
    if expected != actual:
        errors.append("screenshot file set or dimensions changed")
    if requests < 3:
        errors.append("fewer than three complete real load requests")
    return errors


def inventory() -> dict:
    from PIL import Image

    return {
        str(p.relative_to(ROOT)): list(Image.open(p).size)
        for p in sorted((ROOT / "docs/images/console").glob("*/*.webp"))
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / out.stem
    work.mkdir(exist_ok=True)
    home = work / "home"
    home.mkdir(exist_ok=True)
    ep, cp = free_ports(2)
    env = {
        **os.environ,
        "HOME": str(home),
        "TMPDIR": str(work),
        "PYTHONPATH": str(ROOT / "python") + ":" + str(ROOT),
        "YUNSHU_CONSOLE_DB": str(work / "history.sqlite"),
        "YUNSHU_CONSOLE_PORT": str(cp),
        "YUNSHU_VLM_APC_DISK_DIR": str(work / "apc"),
        "YUNSHU_SSD_CACHE_DIR": str(work / "text-cache"),
        "HF_HOME": "/Volumes/P5Plus/hf-cache",
    }
    expected = inventory()
    stop = threading.Event()
    counts = {"complete_requests": 0, "failures": 0}
    url = f"http://127.0.0.1:{ep}"
    with (work / "engine.log").open("wb") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "yunshu_cli",
                "serve",
                "-m",
                args.model,
                "--port",
                str(ep),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        worker = None
        try:
            deadline = time.time() + 600
            while time.time() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError("engine exited; inspect engine.log")
                try:
                    if get(url + "/health/ready", 3) and get(
                        f"http://127.0.0.1:{cp}/v1/yunshu/console", 3
                    ).get("up"):
                        break
                except Exception:
                    time.sleep(2)
            else:
                raise RuntimeError("engine/console startup timeout")
            model = get(url + "/v1/models")["data"][0]["id"]

            def load():
                while not stop.is_set():
                    try:
                        r = post(
                            url + "/v1/chat/completions",
                            {
                                "model": model,
                                "max_tokens": 128,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Explain prefix caching and give a Python example. "
                                        + str(counts["complete_requests"]),
                                    }
                                ],
                            },
                        )
                        if not r.get("choices") or not r.get("usage", {}).get(
                            "completion_tokens", 0
                        ):
                            raise RuntimeError("incomplete load response")
                        counts["complete_requests"] += 1
                    except Exception:
                        counts["failures"] += 1
                    stop.wait(0.3)

            worker = threading.Thread(target=load)
            worker.start()
            time.sleep(15)
            if args.pilot:
                subprocess.run(
                    [
                        "node",
                        "scripts/docs/console_shots.mjs",
                        f"http://127.0.0.1:{cp}",
                        str(work / "pilot"),
                    ],
                    cwd=ROOT,
                    env={**env, "PAGES": "overview", "FAST": "1", "LOCALE": "en-US"},
                    check=True,
                )
            else:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/docs/console_shots.py",
                        f"http://127.0.0.1:{cp}",
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                )
            errors = validate(expected, inventory(), counts["complete_requests"])
            if counts["failures"]:
                errors.append(f"real load failures: {counts['failures']}")
            history = get(f"http://127.0.0.1:{cp}/v1/yunshu/requests/history?limit=100")
            result = {
                "complete": True,
                "pass": not errors,
                "errors": errors,
                **counts,
                "model": args.model,
                "ports": [ep, cp],
                "images": inventory(),
                "history_count": history.get("count"),
                "engine_log": str(work / "engine.log"),
            }
            out.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in result.items() if k != "images"}),
                flush=True,
            )
            return int(bool(errors))
        finally:
            stop.set()
            if worker:
                worker.join(timeout=310)
            stop_tree(proc)


if __name__ == "__main__":
    raise SystemExit(main())
