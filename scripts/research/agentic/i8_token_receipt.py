"""GPU-only correctness replay with raw runner token receipts and a cold/hit pair.

Submit without --quiet. The server-only wrapper observes emitted token IDs;
serving code, prompts, numerical plans and samplers are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PYTHON = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python"


def serve():
    sys.path.insert(0, str(ROOT / "python"))
    from yunshu_engine import vlm_batch_runner as vbr

    emit = vbr.VLMBatchRunner._emit
    receipts = {}

    def observe(self, job, item):
        if not job.terminal:
            key = id(job)
            if isinstance(item, tuple):
                digest, count = receipts.get(key, (hashlib.sha256(), 0))
                digest.update(struct.pack("<I", int(item[0])))
                receipts[key] = digest, count + 1
            elif item is vbr._DONE:
                digest, count = receipts.pop(key, (hashlib.sha256(), 0))
                prompt = b"".join(struct.pack("<I", int(t)) for t in job.ids)
                print(
                    "I8_TOKEN_RECEIPT "
                    + json.dumps(
                        dict(
                            prompt_sha256=hashlib.sha256(prompt).hexdigest(),
                            prompt_tokens=len(job.ids),
                            token_sha256=digest.hexdigest(),
                            tokens=count,
                            cached_tokens=job.stats.cached_tokens,
                            priority=job.priority,
                        )
                    ),
                    flush=True,
                )
        return emit(self, job, item)

    vbr.VLMBatchRunner._emit = observe
    sys.argv = ["yunshu", *sys.argv[2:]]
    from yunshu_cli import main

    raise SystemExit(main())


def main():
    import i8_session_replay as session
    from i8_session_replay import replay
    from servers import Server

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bodies", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bodies = [
        (p.name, json.loads(p.read_text()))
        for p in sorted(args.bodies.glob("*-req.json"))
    ]
    if len(bodies) != 7 or bodies[0][1].get("tools"):
        raise ValueError("expected title plus six main turns")
    if args.dry_run:
        print("validated seven bodies and receipt wrapper imports")
        return
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    command = Server.command

    def receipt_command(server):
        return [PYTHON, str(Path(__file__).resolve()), "--serve", *command(server)[1:]]

    Server.command = receipt_command
    send = session._send
    records, by_arm = [], {}
    for arm in (0, 1):
        os.environ["YUNSHU_AUXILIARY_SCHEDULING"] = str(arm)
        os.environ["YUNSHU_UNCACHED_SCHEDULING"] = str(arm)
        os.environ["AGENTIC_YUNSHU_SRC"] = str(ROOT / "python")
        target = {**bodies[1][1], "seed": 42}
        repeated = False

        def cold_and_hit(url, body):
            nonlocal repeated
            result = send(url, body)
            if body == target and not repeated:
                repeated = True
                warm = send(url, body)
                if result["output_sha256"] != warm["output_sha256"]:
                    raise AssertionError("cold vs hit visible output differs")
            return result

        session._send = cold_and_hit
        prefix = args.out.with_name(f"{args.out.stem}-arm{arm}")
        cfg = SimpleNamespace(
            checkpoint=args.checkpoint,
            cache_dir=prefix.with_suffix(".apc"),
            log=prefix.with_suffix(".server.log"),
            out=prefix.with_suffix(".jsonl"),
            label=f"i8-token-arm{arm}",
        )
        visible = replay(cfg, bodies, 0)
        log = cfg.log.with_name(f"{cfg.log.name}.attempt0")
        raw = [
            json.loads(line.split("I8_TOKEN_RECEIPT ", 1)[1])
            for line in log.read_text().splitlines()
            if line.startswith("I8_TOKEN_RECEIPT ")
        ]
        raw = [r for r in raw if r["prompt_tokens"] > 100]
        if len(raw) != 8:
            raise AssertionError(f"expected eight token receipts, got {len(raw)}")
        grouped = {}
        for row in raw:
            grouped.setdefault(row["prompt_sha256"], []).append(row)
        pairs = [rows for rows in grouped.values() if len(rows) == 2]
        if len(pairs) != 1:
            raise AssertionError("missing repeated cold/hit prompt")
        cold, hit = pairs[0]
        if cold["cached_tokens"] != 0 or hit["cached_tokens"] <= 0:
            raise AssertionError("cold/hit pair did not engage APC")
        if cold["token_sha256"] != hit["token_sha256"]:
            raise AssertionError("cold/hit raw tokens differ")
        by_arm[arm] = {key: rows[0] for key, rows in grouped.items()}
        records.append(dict(kind="arm", arm=arm, raw=raw, visible=visible))
    if by_arm[0].keys() != by_arm[1].keys():
        raise AssertionError("different prompt tokens across arms")
    for key, before in by_arm[0].items():
        after = by_arm[1][key]
        if (before["token_sha256"], before["tokens"]) != (
            after["token_sha256"],
            after["tokens"],
        ):
            raise AssertionError(f"scheduling raw token mismatch: {key}")
    records.append(dict(kind="complete", token_pairs=7, cold_hit_pairs=2))
    args.out.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps(records[-1]), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--serve"]:
        serve()
    else:
        main()
