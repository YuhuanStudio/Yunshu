"""Replay a complete captured opencode session with its independent title request.

Unlike the contention microbenchmark, send the title once and never await it
between main turns. Preserve all six captured main prompts / tool histories and
original output limits. Seed 42 makes complete output comparisons decisive.
GPU-only: submit through gpuq at priority -1.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from replay_traffic import send
from servers import Server, free_ports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bodies", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    requests = sorted(args.bodies.glob("*-req.json"))
    bodies = [(path.name, json.loads(path.read_text())) for path in requests]
    if (
        len(bodies) != 7
        or bodies[0][1].get("tools")
        or any(not b.get("tools") for _, b in bodies[1:])
    ):
        raise ValueError(
            "expected one captured title followed by six captured agent turns"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.cache_dir.exists():
        raise FileExistsError(f"cold replay requires a new APC root: {args.cache_dir}")
    server = Server(
        "yunshu",
        args.checkpoint,
        free_ports(1)[0],
        args.log,
        extra=("--set", f"YUNSHU_VLM_APC_DISK_DIR={args.cache_dir}"),
    ).start()

    def emit(kind, name, result):
        record = dict(kind=kind, body=name, label=args.label, seed=42, **result)
        with args.out.open("a") as f:
            f.write(json.dumps(record) + "\n")
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

    def body(original):
        return {**original, "seed": 42}

    try:
        for _ in range(2):
            send(
                server.url,
                dict(
                    model=bodies[0][1]["model"],
                    messages=[dict(role="user", content="Say hi.")],
                    max_tokens=24,
                ),
            )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(send, server.url, body(bodies[0][1]))
            time.sleep(0.3)
            for name, original in bodies[1:]:
                emit("main", name, send(server.url, body(original)))
            emit("title", bodies[0][0], future.result())
        emit("complete", "", {})
    finally:
        server.kill()


if __name__ == "__main__":
    main()
