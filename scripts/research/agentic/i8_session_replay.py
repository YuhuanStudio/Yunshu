"""Replay a captured opencode session, retrying whole sessions affected by gpuq pauses.

The title is sent once, independently of all six main turns. Preserve captured
prompts, histories and output limits; use seed 42 for complete output parity.
GPU-only: submit through gpuq at priority -1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))
from gpuq_pause import was_paused  # noqa: E402
from replay_traffic import _send  # noqa: E402
from servers import Server, free_ports  # noqa: E402


def replay(args, bodies, attempt):
    cache = args.cache_dir.with_name(f"{args.cache_dir.name}-attempt{attempt}")
    if cache.exists():
        raise FileExistsError(f"cold replay requires a new APC root: {cache}")
    log = args.log.with_name(f"{args.log.name}.attempt{attempt}")
    server = Server(
        "yunshu",
        args.checkpoint,
        free_ports(1)[0],
        log,
        extra=("--set", f"YUNSHU_VLM_APC_DISK_DIR={cache}"),
    )
    records = []

    def emit(kind, name, result):
        record = dict(kind=kind, body=name, label=args.label, seed=42, **result)
        records.append(record)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if k not in ("text", "content", "reasoning")
                }
            ),
            flush=True,
        )

    try:
        # gpuq enforces active-time timeout / stall. A SIGSTOP may last hours;
        # an ordinary 600s wall-clock readiness timeout would reject a healthy
        # server when resumed. No measured sample is accepted across a pause.
        server.start(ready_timeout=86_400)
        for _ in range(2):
            _send(
                server.url,
                dict(
                    model=bodies[0][1]["model"],
                    messages=[dict(role="user", content="Say hi.")],
                    max_tokens=24,
                ),
            )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_send, server.url, {**bodies[0][1], "seed": 42})
            time.sleep(0.3)
            for name, original in bodies[1:]:
                emit("main", name, _send(server.url, {**original, "seed": 42}))
            emit("title", bodies[0][0], future.result())
        emit("complete", "", {})
        return records
    finally:
        server.kill()
        if records:
            raw = args.out.with_name(f"{args.out.name}.attempt{attempt}")
            raw.write_text("".join(json.dumps(r) + "\n" for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bodies", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    bodies = [
        (path.name, json.loads(path.read_text()))
        for path in sorted(args.bodies.glob("*-req.json"))
    ]
    if (
        len(bodies) != 7
        or bodies[0][1].get("tools")
        or any(not b.get("tools") for _, b in bodies[1:])
    ):
        raise ValueError("expected one captured title and six captured agent turns")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    for attempt in range(5):
        t0 = time.time()
        try:
            records = replay(args, bodies, attempt)
        except Exception:
            if not was_paused(t0, time.time()):
                raise
            print(
                f"session {args.label} attempt {attempt}: paused; retrying fresh server",
                flush=True,
            )
            continue
        if was_paused(t0, time.time()):
            print(
                f"session {args.label} attempt {attempt}: paused; retrying fresh server",
                flush=True,
            )
            continue
        args.out.write_text("".join(json.dumps(r) + "\n" for r in records))
        return
    raise RuntimeError("five paused session attempts; no valid measurement accepted")


if __name__ == "__main__":
    main()
