"""Report latency and task success per case from bench_user_latency JSONL."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def quantile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)] if values else None


def metric(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return {
        "n": len(values),
        "median_s": round(statistics.median(values), 3),
        "p95_s": round(quantile(values, 0.95), 3),
        "max_s": round(max(values), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    for line in args.path.open():
        if line.strip():
            row = json.loads(line)
            groups[row["case"]].append(row)
    for case, rows in groups.items():
        print(
            json.dumps(
                {
                    "case": case,
                    "requests": len(rows),
                    "task_ok": sum(row.get("task_ok") is True for row in rows),
                    "http_ok": sum(
                        row.get("http_status") == 200 and row.get("done_received")
                        for row in rows
                    ),
                    "first_content": metric(rows, "first_content_s"),
                    "first_reasoning": metric(rows, "first_reasoning_s"),
                    "complete": metric(rows, "wall_s"),
                    "cached_tokens": [
                        row.get("usage", {})
                        .get("prompt_tokens_details", {})
                        .get("cached_tokens")
                        for row in rows
                        if isinstance(row.get("usage"), dict)
                    ],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
