"""Summarize an HTTP soak trace without interpreting retained memory as a leak.

Works on an incomplete JSONL too; the output says whether idle_end was seen.
MMLU-Pro letter matching is a smoke signal, not an official accuracy score.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def gib(value):
    return round(value / 1024**3, 3) if value is not None else None


def series_summary(rows, get_value):
    points = [(r.get("elapsed_s"), get_value(r)) for r in rows]
    points = [(t, v) for t, v in points if isinstance(t, (int, float)) and isinstance(v, (int, float))]
    if not points:
        return None
    return {
        "first_gib": gib(points[0][1]),
        "last_gib": gib(points[-1][1]),
        "peak_gib": gib(max(v for _, v in points)),
        "first_elapsed_s": round(points[0][0], 2),
        "last_elapsed_s": round(points[-1][0], 2),
        "samples": len(points),
    }


def summarize(path):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    events = Counter(row.get("event") for row in rows)
    requests = [row for row in rows if row.get("event") == "request"]
    memory = [row for row in rows if row.get("event") == "memory"]
    successful = [
        row
        for row in requests
        if row.get("http_status") == 200
        and row.get("done_received")
        and not row.get("error")
        and not row.get("stream_error")
    ]
    lengths = [row.get("wall_s") for row in requests if isinstance(row.get("wall_s"), (int, float))]
    workload = next((row.get("arguments", {}).get("workload") for row in rows if row.get("event") == "start"), None)
    result = {
        "path": str(path),
        "workload": workload,
        "complete": bool(events["idle_end"]),
        "events": dict(events),
        "request_count": len(requests),
        "success_count": len(successful),
        "http_statuses": dict(Counter(str(row.get("http_status")) for row in requests)),
        "finish_reasons": dict(Counter(str(row.get("finish_reason")) for row in requests)),
        "error_count": sum(bool(row.get("error") or row.get("stream_error")) for row in requests),
        "guard_count": sum(bool(row.get("guard_triggered")) for row in memory),
        "request_wall_s": {
            "median": round(statistics.median(lengths), 3) if lengths else None,
            "p95": round(percentile(lengths, 0.95), 3) if lengths else None,
            "max": round(max(lengths), 3) if lengths else None,
        },
        "process_tree_rss": series_summary(
            memory, lambda row: row.get("process_memory", {}).get("rss_sum_bytes")
        ),
        "process_tree_phys_footprint": series_summary(
            memory,
            lambda row: row.get("process_memory", {}).get("physical_footprint_sum_bytes"),
        ),
        "engine_current": series_summary(
            memory,
            lambda row: row.get("engine_status", {}).get("memory_actual", {}).get("current_bytes"),
        ),
    }
    if workload == "mixed":
        result["exact_output_count"] = sum(row.get("exact_match") is True for row in requests)
    elif workload == "mmlu-pro":
        answers = [
            row for row in successful if row.get("finish_reason") == "stop" and row.get("output", "").strip() in set("ABCDEFGHIJ")
        ]
        result["single_letter_output_count"] = len(answers)
        result["sampled_letter_match_count"] = sum(
            row["output"].strip() == row.get("expected_answer") for row in answers
        )
        result["length_limited_count"] = sum(row.get("finish_reason") == "length" for row in requests)
        result["sampled_letter_match_warning"] = "Exploratory output check only; not official MMLU-Pro accuracy."
    idle_start = next((row["elapsed_s"] for row in rows if row.get("event") == "idle_start"), None)
    if idle_start is not None:
        idle_rows = [row for row in memory if row.get("elapsed_s", 0) >= idle_start]
        result["idle_duration_observed_s"] = round(memory[-1]["elapsed_s"] - idle_start, 2) if idle_rows else 0
        result["idle_engine_current"] = series_summary(
            idle_rows,
            lambda row: row.get("engine_status", {}).get("memory_actual", {}).get("current_bytes"),
        )
        result["idle_phys_footprint"] = series_summary(
            idle_rows,
            lambda row: row.get("process_memory", {}).get("physical_footprint_sum_bytes"),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        print(json.dumps(summarize(path), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
