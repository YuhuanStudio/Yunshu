"""Bounded real HTTP soak with process RSS and optional engine status telemetry.

This is a workload trace, not proof of leak freedom. A sampled MMLU-Pro run is
not the official benchmark; no accuracy claim without an explicit scorer.
"""

import argparse
import hashlib
import json
import random
import threading
import time
import urllib.request
from pathlib import Path

from process_memory import process_tree_memory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--status-path", default="/status")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workload", choices=["mixed", "mmlu-pro"], default="mixed")
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("reference/omlx/omlx/eval/data/mmlu_pro_test.jsonl"),
    )
    p.add_argument("--requests", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--idle-seconds", type=int, default=60)
    p.add_argument("--rss-stop-gib", type=float, default=64)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    stop = threading.Event()
    guard = threading.Event()
    origin = time.monotonic()

    def record(d):
        d["elapsed_s"] = time.monotonic() - origin
        with lock, a.output.open("a") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def sample():
        while not stop.is_set():
            row = {"event": "memory"}
            try:
                row["process_memory"] = process_tree_memory(a.pid)
                observed = max(
                    row["process_memory"]["rss_sum_bytes"],
                    row["process_memory"]["physical_footprint_sum_bytes"],
                )
                if observed > a.rss_stop_gib * 1024**3:
                    guard.set()
                    row["guard_triggered"] = True
            except Exception as e:
                row["rss_error"] = str(e)
            try:
                with urllib.request.urlopen(a.url + a.status_path, timeout=2) as r:
                    row["engine_status"] = json.load(r)
                    if (
                        row["engine_status"]
                        .get("memory_actual", {})
                        .get("current_bytes", 0)
                        > a.rss_stop_gib * 1024**3
                    ):
                        guard.set()
                        row["guard_triggered"] = True
            except Exception as e:
                row["status_error"] = str(e)
            record(row)
            stop.wait(1)

    items = []
    if a.workload == "mmlu-pro":
        raw = a.dataset.read_bytes()
        pool = [json.loads(l) for l in raw.splitlines()]
        random.Random(20260927).shuffle(pool)
        items = pool[: a.requests]
        record(
            dict(
                event="dataset",
                path=str(a.dataset),
                sha256=hashlib.sha256(raw).hexdigest(),
                sample_ids=[x["id"] for x in items],
                sampling="seeded shuffle 20260927; not official benchmark",
            )
        )
    record(
        dict(
            event="start",
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
            },
        )
    )
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    try:
        for i in range(a.requests):
            if guard.is_set():
                record(
                    dict(event="stopped", reason="RSS safety threshold", completed=i)
                )
                break
            if a.workload == "mmlu-pro":
                x = items[i]
                text = (
                    "Answer the following question. Answer with just the letter.\n\nQuestion: "
                    + x["question"]
                    + "\n\n"
                    + "\n".join(
                        label + ". " + choice
                        for label, choice in zip(x["labels"], x["choices"], strict=True)
                    )
                    + "\n\nAnswer:"
                )
                meta = {
                    "dataset_id": x["id"],
                    "subject": x["subject"],
                    "expected_answer": x["answer"],
                }
            else:
                # Alternates reusable and unique prefixes; output requires recall
                # near the end. Shared prefix alone would underexercise eviction.
                key = i if i % 2 else i % 4
                text = (
                    f"Document {key}: archived account records have no bearing on the final code.\n"
                    * 300
                ) + f"\nThe final code is CODE{i:04d}. Return only this code."
                meta = {"expected_answer": f"CODE{i:04d}", "prefix_key": key}
            body = dict(
                model=a.model,
                messages=[{"role": "user", "content": text}],
                temperature=0,
                max_tokens=a.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                chat_template_kwargs={"enable_thinking": a.thinking},
                reasoning_effort="medium" if a.thinking else "none",
            )
            row = dict(
                event="request",
                index=i,
                **meta,
                input_sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
            started = time.monotonic()
            first = None
            output = []
            reasoning = []
            usage = None
            try:
                req = urllib.request.Request(
                    a.url + "/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=600) as r:
                    row["http_status"] = r.status
                    for line in r:
                        if not line.startswith(b"data: "):
                            continue
                        if line[6:].strip() == b"[DONE]":
                            row["done_received"] = True
                            break
                        data = json.loads(line[6:])
                        if data.get("error"):
                            row["stream_error"] = data["error"]
                        if data.get("usage"):
                            usage = data["usage"]
                        for c in data.get("choices", []):
                            d = c.get("delta", {})
                            if d.get("content"):
                                if first is None:
                                    first = time.monotonic() - started
                                output.append(d["content"])
                            if d.get("reasoning_content"):
                                reasoning.append(d["reasoning_content"])
                            if c.get("finish_reason"):
                                row["finish_reason"] = c["finish_reason"]
                        if guard.is_set():
                            row["aborted_by_guard"] = True
                            break
                row.update(
                    output="".join(output),
                    reasoning="".join(reasoning),
                    usage=usage,
                    first_text_s=first,
                    wall_s=time.monotonic() - started,
                )
                if a.workload == "mixed":
                    row["exact_match"] = (
                        row["output"].strip() == meta["expected_answer"]
                    )
            except Exception as e:
                row.update(error=repr(e), wall_s=time.monotonic() - started)
                if hasattr(e, "read"):
                    row["error_body"] = e.read().decode(errors="replace")
            record(row)
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ["output", "reasoning"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        record(dict(event="idle_start", duration_s=a.idle_seconds))
        # Telemetry continues during idle; user-facing orchestration remains free.
        stop.wait(a.idle_seconds)
        record(dict(event="idle_end"))
    finally:
        stop.set()
        t.join(timeout=5)


if __name__ == "__main__":
    main()
