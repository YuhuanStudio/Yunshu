"""Capture the public console images under real load. Run only through gpuq."""

from __future__ import annotations

import argparse
import json
import os
import re
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


def swap_mb(text: str) -> float:
    match = re.search(r"used\s*=\s*([\d.]+)([MG])", text)
    if not match:
        raise ValueError("cannot parse host swap")
    return float(match[1]) * (1024 if match[2] == "G" else 1)


def validate_long_usage(usage: dict, cached: bool) -> list[str]:
    errors = []
    if usage.get("prompt_tokens", 0) < 32768:
        errors.append("context below 32K")
    if usage.get("completion_tokens", 0) != 2048:
        errors.append("long reply incomplete")
    if (
        cached
        and usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) < 32768
    ):
        errors.append("32K cached prefix not engaged")
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
    ap.add_argument(
        "--context-file",
        type=Path,
        help="32K+ exact-token corpus prompt; sequential long agent session",
    )
    ap.add_argument("--max-host-swap-mb", type=float)
    args = ap.parse_args()
    initial_swap = swap_mb(
        subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    )
    if args.max_host_swap_mb is not None and initial_swap > args.max_host_swap_mb:
        raise RuntimeError(
            f"capture requires low host swap: {initial_swap} MiB exceeds {args.max_host_swap_mb} MiB"
        )
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
    if args.context_file:
        env["YUNSHU_VLM_DRAFT"] = "mtp"
    expected = inventory()
    primed = threading.Event()
    usage_rows = []
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
                messages = []
                if args.context_file:
                    messages = [
                        {"role": "user", "content": args.context_file.read_text()}
                    ]
                while not stop.is_set():
                    try:
                        body = {
                            "model": model,
                            "max_tokens": 2048 if args.context_file else 128,
                            "temperature": 0,
                            "chat_template_kwargs": {"enable_thinking": False},
                            "messages": messages
                            if args.context_file
                            else [
                                {
                                    "role": "user",
                                    "content": "Explain prefix caching and give a Python example. "
                                    + str(counts["complete_requests"]),
                                }
                            ],
                        }
                        r = post(url + "/v1/chat/completions", body, timeout=900)
                        usage = r.get("usage", {})
                        if not r.get("choices") or not usage.get(
                            "completion_tokens", 0
                        ):
                            raise RuntimeError("incomplete load response")
                        if args.context_file:
                            errors = validate_long_usage(
                                usage, counts["complete_requests"] > 0
                            )
                            if errors:
                                raise RuntimeError("; ".join(errors))
                            # Metadata only: never persist prompt/completion content.
                            usage_rows.append(usage)
                            if counts["complete_requests"] > 0:
                                messages.extend(
                                    [
                                        {
                                            "role": "assistant",
                                            "content": r["choices"][0]["message"][
                                                "content"
                                            ],
                                        },
                                        {
                                            "role": "user",
                                            "content": "Continue implementing the next scheduler component with detailed Python code, tests and reasoning. Write at least 3000 words.",
                                        },
                                    ]
                                )
                        counts["complete_requests"] += 1
                        if counts["complete_requests"] >= 3:
                            primed.set()
                    except Exception as exc:
                        print("load failed:", str(exc), flush=True)
                        counts["failures"] += 1
                        primed.set()
                        return
                    if not args.context_file:
                        stop.wait(0.3)

            worker = threading.Thread(target=load)
            worker.start()
            if args.context_file:
                if not primed.wait(1200) or counts["failures"]:
                    raise RuntimeError("long session did not prime successfully")
            else:
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
            final_swap = swap_mb(
                subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
            )
            errors = validate(expected, inventory(), counts["complete_requests"])
            if args.max_host_swap_mb is not None and final_swap > args.max_host_swap_mb:
                errors.append("host swap too high at capture end")
            if args.context_file and not re.search(
                r"VLM batch runner: [^\n]*draft=mtp\b",
                (work / "engine.log").read_text(errors="replace"),
            ):
                errors.append("MTP engagement missing from engine log")
            if counts["failures"]:
                errors.append(f"real load failures: {counts['failures']}")
            history = get(f"http://127.0.0.1:{cp}/v1/yunshu/requests/history?limit=100")
            result = {
                "complete": True,
                "pass": not errors,
                "errors": errors,
                **counts,
                "model": args.model,
                "long_session_usage": usage_rows,
                "host_swap_mb_start": initial_swap,
                "host_swap_mb_end": final_swap,
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
