"""Pair MTP off/on soak traces by fixed MMLU-Pro sample IDs.

Reports task latency, output changes and memory. It does not compute official
MMLU-Pro accuracy or infer MTP engagement from the command-line toggle.
"""

import argparse
import json
import re
import statistics
from pathlib import Path


def load(path):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    dataset = next(row for row in rows if row.get("event") == "dataset")
    requests = [row for row in rows if row.get("event") == "request"]
    memory = [row for row in rows if row.get("event") == "memory"]
    start = next(row for row in rows if row.get("event") == "start")
    return dataset, start, requests, memory


def median(values):
    return round(statistics.median(values), 3) if values else None


def memory_peak(rows):
    return round(
        max(
            (row.get("process_memory", {}).get("physical_footprint_sum_bytes", 0) for row in rows),
            default=0,
        )
        / 1024**3,
        3,
    )


def parse_mtp_log(path):
    if path is None:
        return None
    pattern = re.compile(r"MTP\[[^]]+\] finish=\S+ tokens=(\d+) cycles=(\d+) tok/cycle=([0-9.]+) accept=(\d+)/(\d+)")
    records = [tuple(map(float, match.groups())) for match in pattern.finditer(path.read_text(errors="replace"))]
    accepted = sum(row[3] for row in records)
    drafted = sum(row[4] for row in records)
    return {
        "path": str(path),
        "summaries": len(records),
        "total_generated_tokens": int(sum(row[0] for row in records)),
        "accepted_draft_tokens": int(accepted),
        "drafted_tokens": int(drafted),
        "acceptance_fraction": round(accepted / drafted, 4) if drafted else None,
        "median_tokens_per_cycle": median([row[2] for row in records]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("off", type=Path)
    parser.add_argument("on", type=Path)
    parser.add_argument("--mtp-server-log", type=Path)
    args = parser.parse_args()
    off_data, off_start, off_rows, off_mem = load(args.off)
    on_data, on_start, on_rows, on_mem = load(args.on)
    if off_data["sha256"] != on_data["sha256"] or off_data["sample_ids"] != on_data["sample_ids"]:
        parser.error("Dataset hash or sampled question order differs")
    off_args = off_start["arguments"]
    on_args = on_start["arguments"]
    for key in ("workload", "requests", "max_tokens", "thinking", "idle_seconds"):
        if off_args.get(key) != on_args.get(key):
            parser.error(f"Control condition differs: {key}")
    off_by_id = {row["dataset_id"]: row for row in off_rows}
    on_by_id = {row["dataset_id"]: row for row in on_rows}
    paired = [
        (off_by_id[question], on_by_id[question])
        for question in off_data["sample_ids"]
        if question in off_by_id and question in on_by_id
    ]
    timings = [
        (a["wall_s"], b["wall_s"])
        for a, b in paired
        if isinstance(a.get("wall_s"), (int, float))
        and isinstance(b.get("wall_s"), (int, float))
        and a.get("http_status") == b.get("http_status") == 200
        and a.get("done_received")
        and b.get("done_received")
    ]
    completed_answer_pairs = [
        (a, b)
        for a, b in paired
        if a.get("http_status") == b.get("http_status") == 200
        and a.get("done_received")
        and b.get("done_received")
        and a.get("finish_reason") == b.get("finish_reason") == "stop"
        and a.get("output", "").strip()
        and b.get("output", "").strip()
        and isinstance(a.get("wall_s"), (int, float))
        and isinstance(b.get("wall_s"), (int, float))
    ]
    token_rates = [
        (
            a.get("usage", {}).get("generation_tokens_per_second"),
            b.get("usage", {}).get("generation_tokens_per_second"),
        )
        for a, b in paired
        if isinstance(a.get("usage"), dict) and isinstance(b.get("usage"), dict)
    ]
    first_text = [
        (a["first_text_s"], b["first_text_s"])
        for a, b in paired
        if isinstance(a.get("first_text_s"), (int, float))
        and isinstance(b.get("first_text_s"), (int, float))
    ]
    first_visible = [
        (
            a["usage"]["time_to_first_visible_token"],
            b["usage"]["time_to_first_visible_token"],
        )
        for a, b in paired
        if isinstance(a.get("usage"), dict)
        and isinstance(b.get("usage"), dict)
        and isinstance(a["usage"].get("time_to_first_visible_token"), (int, float))
        and isinstance(b["usage"].get("time_to_first_visible_token"), (int, float))
    ]
    output_changes = [
        {
            "id": a["dataset_id"],
            "off_output": a.get("output", "").strip(),
            "on_output": b.get("output", "").strip(),
            "expected": a.get("expected_answer"),
            "off_finish": a.get("finish_reason"),
            "on_finish": b.get("finish_reason"),
        }
        for a, b in paired
        if a.get("output", "").strip() != b.get("output", "").strip()
    ]
    result = {
        "off_path": str(args.off),
        "on_path": str(args.on),
        "dataset_sha256": off_data["sha256"],
        "planned_questions": len(off_data["sample_ids"]),
        "off_completed": len(off_rows),
        "on_completed": len(on_rows),
        "paired_successful": len(timings),
        "off_median_wall_s": median([a for a, _ in timings]),
        "on_median_wall_s": median([b for _, b in timings]),
        "paired_median_speedup": median([a / b for a, b in timings if b > 0]),
        "both_completed_answer_count": len(completed_answer_pairs),
        "both_completed_answer_median_speedup": median(
            [a["wall_s"] / b["wall_s"] for a, b in completed_answer_pairs if b["wall_s"] > 0]
        ),
        "off_empty_final_count": sum(not row.get("output", "").strip() for row in off_rows),
        "on_empty_final_count": sum(not row.get("output", "").strip() for row in on_rows),
        "off_length_count": sum(row.get("finish_reason") == "length" for row in off_rows),
        "on_length_count": sum(row.get("finish_reason") == "length" for row in on_rows),
        "off_median_first_content_s": median([a for a, _ in first_text]),
        "on_median_first_content_s": median([b for _, b in first_text]),
        "off_median_server_first_visible_token_s": median([a for a, _ in first_visible]),
        "on_median_server_first_visible_token_s": median([b for _, b in first_visible]),
        "off_median_reported_generation_tps": median([a for a, _ in token_rates if isinstance(a, (int, float))]),
        "on_median_reported_generation_tps": median([b for _, b in token_rates if isinstance(b, (int, float))]),
        "off_peak_phys_footprint_gib": memory_peak(off_mem),
        "on_peak_phys_footprint_gib": memory_peak(on_mem),
        "mtp_runtime_evidence": parse_mtp_log(args.mtp_server_log),
        "output_change_count": len(output_changes),
        "output_changes": output_changes,
        "limits": "Check actual MTP draft/verify logs and cache/temperature settings separately; completion and first-content time include different reasoning/output lengths. Server first-visible-token is not socket first content.",
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
