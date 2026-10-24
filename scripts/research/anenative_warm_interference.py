"""Decode interference with one server and persistent paused/resumed ANE workers.

APC is RAM-hot by design in every measured arm; no TTFT claim. This avoids
summing compressed-page peaks across repeated server/worker lifetimes and avoids
unrequested 30-second idle probes. Console is explicitly off; ports18990..18996.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from anenative_tfbench import metrics_only


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--server-src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--channels", type=int, default=2176)
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--mode", choices=["fp16", "int8"], default="fp16")
    p.add_argument("--worker-smoke", action="store_true")
    return p


def validate_row(row, kind, cache_by_kind, sha_by_kind):
    if row.get("ct") != 512 or row.get("finish") != "length":
        raise ValueError("incomplete decode")
    if not isinstance(row.get("cached"), int) or row["cached"] <= 0:
        raise ValueError("expected RAM-hot prefix in every measured arm")
    if row.get("sha") != sha_by_kind[kind]:
        raise ValueError("warm/background output differs from cold prime")
    if row["xy"]["speculative"]["mode"] != "dflash":
        raise ValueError("request did not engage DFlash")
    if kind in cache_by_kind and row["cached"] != cache_by_kind[kind]:
        raise ValueError("unequal cached prefix across arms")
    cache_by_kind[kind] = row["cached"]


def main():
    a = parser().parse_args()
    if a.channels <= 0 or a.rows <= 0 or not (a.server_src / "python").is_dir():
        raise ValueError("invalid shape or server source")
    a.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    ane_python = "/Volumes/P5Plus/venvs/anesplit-mlx/bin/python"
    workers = {}
    logs = []
    server = None
    apc = Path("/Volumes/Micron/yunshu-apc") / f"anenative-warm-{a.tag}"

    def pause_all():
        for proc in workers.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGUSR2)

    try:
        for backend in ["native", "coreml"]:
            ready = a.out / f"{backend}.ready"
            ready.unlink(missing_ok=True)
            log = (a.out / f"{backend}.log").open("w")
            logs.append(log)
            proc = subprocess.Popen(
                [
                    ane_python,
                    str(root / "scripts/research/anenative_load.py"),
                    "--backend",
                    backend,
                    "--rows",
                    str(a.rows),
                    "--channels",
                    str(a.channels),
                    "--mode",
                    a.mode,
                    "--start-paused",
                    "--cache",
                    str(a.out / "cache"),
                    "--ready",
                    str(ready),
                    "--out",
                    str(a.out / f"{backend}.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": str(root / "python")},
            )
            workers[backend] = proc
            deadline = time.monotonic() + 240
            while not ready.exists():
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        f"{backend} worker failed to prepare; see its log"
                    )
                time.sleep(0.05)
            assert json.loads(ready.read_text())["active"] is False
        if a.worker_smoke:
            for backend in ["native", "coreml"]:
                workers[backend].send_signal(signal.SIGUSR1)
                time.sleep(0.5)
                workers[backend].send_signal(signal.SIGUSR2)
                time.sleep(0.25)
        else:
            os.environ["TFB_YUNSHU_SRC"] = str(a.server_src / "python")
            os.environ["TFB_OUT"] = str(a.out / "tfb")
            import tfbench as t

            t.YUNSHU_SRC = str(a.server_src / "python")
            t.OUT = a.out / "tfb"
            env = {
                "YUNSHU_CONSOLE": "0",
                "YUNSHU_VLM_APC_DISK": "0",
                "YUNSHU_VLM_APC_DISK_DIR": str(apc),
                "YUNSHU_NEXT_TURN_PREFILL": "off",
                "YUNSHU_VLM_DRAFT": "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2",
            }
            server = t.start_server("yunshu", env, "anenative-warm-" + a.tag, model=t.M)
            if server.engaged_spec_mode != "dflash":
                raise RuntimeError("DFlash engagement not proven")
            prompts = {
                kind: t.decode_prompt(kind, 32768, True) for kind in ["prose", "code"]
            }
            cache_by_kind, sha_by_kind = {}, {}
            with (a.out / "decode.jsonl").open("w") as output:

                def emit(record):
                    record = metrics_only(record)
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    print(
                        json.dumps({k: v for k, v in record.items() if k != "xy"})[
                            :500
                        ],
                        flush=True,
                    )

                emit(
                    {
                        "kind": "meta",
                        "server_src": str(a.server_src),
                        "worker_src": str(root),
                        "ctx": 32768,
                        "channels": a.channels,
                        "rows": a.rows,
                        "mode": a.mode,
                        "RAM_hot": True,
                        "APC_disk": False,
                        "rehearsal": "off",
                        "console": False,
                        "device": "m5",
                        "engaged_spec_mode": server.engaged_spec_mode,
                    }
                )
                for kind, prompt in prompts.items():
                    row = t.send(
                        server.url,
                        t.req(server.model, prompt, 512),
                        capture_timestamps=True,
                    )
                    if row.get("ct") != 512 or row.get("finish") != "length":
                        raise ValueError("incomplete cold prime")
                    sha_by_kind[kind] = row["sha"]
                    emit({"stage": "prime", "kind": kind, **row})
                arms = [
                    "base",
                    "cand",
                    "cand-r1",
                    "base-r1",
                    "coreml",
                    "coreml-r1",
                    "base-r2",
                    "cand-r2",
                    "coreml-r2",
                    "base-r3",
                ]
                for arm in arms:
                    pause_all()
                    backend = (
                        "native"
                        if arm.startswith("cand")
                        else "coreml"
                        if arm.startswith("coreml")
                        else "off"
                    )
                    if backend != "off":
                        workers[backend].send_signal(signal.SIGUSR1)
                    time.sleep(0.25)
                    for kind, prompt in prompts.items():
                        row = t.send(
                            server.url,
                            t.req(server.model, prompt, 512),
                            capture_timestamps=True,
                        )
                        row["_end_at_s"] = time.perf_counter()
                        validate_row(row, kind, cache_by_kind, sha_by_kind)
                        if any(proc.poll() is not None for proc in workers.values()):
                            raise RuntimeError(
                                "background load exited during a timing cell"
                            )
                        emit(
                            {
                                "stage": "decode",
                                "arm": arm,
                                "backend": backend,
                                "kind": kind,
                                **row,
                            }
                        )
                pause_all()
                emit({"complete": True, "cached_by_kind": cache_by_kind})
    finally:
        if server is not None:
            server.kill()
        for proc in workers.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in workers.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in logs:
            log.close()
        shutil.rmtree(apc, ignore_errors=True)
    for backend, proc in workers.items():
        record = json.loads((a.out / f"{backend}.json").read_text())
        if (
            proc.returncode != 0
            or record.get("complete") is not True
            or record["calls"] <= 0
        ):
            raise RuntimeError("incomplete worker receipt")
        if a.worker_smoke:
            assert any(
                not s["active"] and s["calls"] == 0 for s in record["rate_samples"]
            )
            assert any(s["active"] and s["calls"] > 0 for s in record["rate_samples"])
        print(
            json.dumps(
                {
                    "backend": backend,
                    "complete": True,
                    "allocator_relief_bytes": record["allocator_relief_bytes"],
                }
            ),
            flush=True,
        )
    (a.out / "complete.json").write_text('{"complete":true}\n')
    if not a.worker_smoke:
        from anenative_warm_summary import summarize

        (a.out / "summary.json").write_text(
            json.dumps(summarize(a.out), indent=2) + "\n"
        )
    print('{"complete":true}', flush=True)


if __name__ == "__main__":
    main()
