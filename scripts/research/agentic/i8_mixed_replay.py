"""Queued paired long-cold / cached-suffix scheduling probe, with complete receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_traffic import _send  # noqa: E402
from servers import Server, free_ports  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def bodies(tiny=False):
    model = "Qwen3.8-27B-oQ4e-mtp"
    title = json.loads(
        (ROOT / "tests/fixtures/auxiliary/opencode_title.json").read_text()
    )
    title["seed"] = 42
    if tiny:
        title["max_tokens"] = 48  # transport smoke; no classifier/parity claim
    # Two unrelated contexts prevent the cold row from donating the warm prefix.
    n = 80 if tiny else 2200
    warm = dict(
        model=model,
        seed=42,
        max_tokens=48,
        messages=[
            dict(
                role="system",
                content="Read the ledger and answer the last instruction.",
            ),
            dict(
                role="user",
                content="Warm ledger\n" + "alpha beta gamma delta\n" * n + "\nSay OK.",
            ),
        ],
    )
    cold = dict(
        model=model,
        seed=42,
        max_tokens=48,
        messages=[
            dict(
                role="system",
                content="Read the archive and answer the last instruction.",
            ),
            dict(
                role="user",
                content="Cold archive\n"
                + "red green blue yellow orange\n" * (n * 2)
                + "\nSay READY.",
            ),
        ],
    )
    suffixes = []
    for count in (32, 512, 2304):
        suffixes.append(
            {
                **warm,
                "messages": [
                    *warm["messages"],
                    dict(role="assistant", content="OK."),
                    dict(role="user", content="suffix " * count + "\nSay DONE."),
                ],
            }
        )
    return title, warm, cold, suffixes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    title, warm, cold, suffixes = bodies(args.tiny)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    requests=2 + len(suffixes),
                    chars=[len(json.dumps(b)) for b in [warm, cold, *suffixes]],
                )
            )
        )
        return
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    os.environ["AGENTIC_YUNSHU_SRC"] = str(ROOT / "python")
    source = {
        name: hashlib.sha256(
            (ROOT / "python/yunshu_engine" / name).read_bytes()
        ).hexdigest()
        for name in (
            "vlm_batch_runner.py",
            "serving/work_scheduler.py",
            "serving/auxiliary_prefill.py",
        )
    }
    all_rows = []
    for run in range(args.rounds):
        for arm in (run % 2, 1 - run % 2):
            os.environ["YUNSHU_AUXILIARY_SCHEDULING"] = str(arm)
            os.environ["YUNSHU_UNCACHED_SCHEDULING"] = str(arm)
            log = args.out / f"{arm}-{run}.server.log"
            cache = args.out / f"cache-{arm}-{run}"
            server = Server(
                "yunshu",
                args.checkpoint,
                free_ports(1)[0],
                log,
                extra=("--set", f"YUNSHU_VLM_APC_DISK_DIR={cache}"),
            )
            rows = []

            def record(kind, result):
                row = dict(kind=kind, arm=arm, run=run, **result)
                rows.append(row)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in ("text", "content", "reasoning")
                        }
                    ),
                    flush=True,
                )

            try:
                server.start(ready_timeout=86400)
                record("prime", _send(server.url, warm))
                with ThreadPoolExecutor(max_workers=2) as pool:
                    aux = pool.submit(_send, server.url, title)
                    time.sleep(0.3)
                    long = pool.submit(_send, server.url, cold)
                    time.sleep(0.3)
                    for i, body in enumerate(suffixes):
                        record(f"suffix-{i}", _send(server.url, body))
                    record("cold", long.result())
                    record("title", aux.result())
                if not args.tiny and "Speculative decoding: mtp" not in log.read_text():
                    raise AssertionError("missing checkpoint MTP engagement")
                if not args.tiny and any(
                    r["cached_tokens"] <= 0
                    for r in rows
                    if r["kind"].startswith("suffix")
                ):
                    raise AssertionError("suffix did not hit APC")
                record(
                    "complete",
                    dict(
                        source_sha256=source, engaged_mode="off" if args.tiny else "mtp"
                    ),
                )
            finally:
                server.kill()
                (args.out / f"{arm}-{run}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
            all_rows.extend(rows)
    pairs = []
    for run in range(args.rounds):
        before = {
            r["kind"]: r
            for r in all_rows
            if r["run"] == run and r["arm"] == 0 and r["kind"] != "complete"
        }
        after = {
            r["kind"]: r
            for r in all_rows
            if r["run"] == run and r["arm"] == 1 and r["kind"] != "complete"
        }
        for kind, row in before.items():
            pairs.append(
                dict(
                    run=run,
                    kind=kind,
                    equal=row["output_sha256"] == after[kind]["output_sha256"],
                )
            )
    summary = dict(
        kind="complete",
        pairs=pairs,
        equal=sum(p["equal"] for p in pairs),
        total=len(pairs),
    )
    (args.out / "complete.jsonl").write_text(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)
    if not args.tiny and summary["equal"] != summary["total"]:
        raise SystemExit("mixed output parity failed")


if __name__ == "__main__":
    main()
