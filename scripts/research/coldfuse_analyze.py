"""Fail-closed CPU analysis of cold-prefill diagnostic and tfbench rows."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def validate_cold(rows, *, expected_tokens=1):
    if not rows or rows[-1].get("complete") is not True:
        raise ValueError("missing final complete record")
    cells = [r for r in rows if r.get("part") == "decode"]
    if not cells:
        raise ValueError("missing decode cells")
    for row in cells:
        if row.get("phase") != "cold" or row.get("cached") != 0:
            raise ValueError("not a genuine cold cell")
        if row.get("ct") != expected_tokens or row.get("finish") != "length":
            raise ValueError("incomplete first-token cell")
        if row.get("device") != "m5" or row.get("ttft_s", 0) <= 0:
            raise ValueError("missing M5 timing")
    return cells


def compare_arms(base, cand, *, require_logprobs=True):
    """Compare cold cells from the same M5 workload, including exact logprobs."""

    def index(rows):
        result = {}
        for row in rows:
            key = (row["ctx"], row["kind"])
            if key in result:
                raise ValueError("duplicate comparison cell")
            if row.get("device") != "m5" or row.get("cached") != 0:
                raise ValueError("comparison must be cold on M5")
            if not row.get("sha") or row.get("ct", 0) <= 0:
                raise ValueError("missing output digest or token count")
            if require_logprobs and (
                not row.get("lp_tokens")
                or not row.get("lps")
                or len(row["lp_tokens"]) != row["ct"]
                or len(row["lps"]) != row["ct"]
            ):
                raise ValueError("missing full token/logprob evidence")
            result[key] = row
        return result

    b, c = index(base), index(cand)
    if not b or b.keys() != c.keys():
        raise ValueError("comparison workloads differ")
    result = []
    for key in sorted(b):
        x, y = b[key], c[key]
        if any(x.get(k) != y.get(k) for k in ("ct", "sha", "lp_tokens", "lps")):
            raise ValueError(f"output or logprob mismatch: {key}")
        result.append(
            dict(
                ctx=key[0],
                kind=key[1],
                base_s=x["ttft_s"],
                cand_s=y["ttft_s"],
                saved_s=x["ttft_s"] - y["ttft_s"],
            )
        )
    return result


def summarize(records, *, compute_peak=65e12, bandwidth_peak=614e9):
    if not records or any(r["cached"] != [0] for r in records):
        raise ValueError("missing profile or non-cold profile")
    totals = collections.defaultdict(lambda: [0.0, 0, 0])
    wall = busy = gaps = 0.0
    for record in records:
        wall += record["wall_s"]
        busy += record["timeline"]["busy_union_ms"] / 1000
        gaps += record["timeline"]["idle_between_cb_ms"] / 1000
        for name, span in record["spans"].items():
            totals[name][0] += span["exclusive_s"]
            totals[name][1] += span["count"]
            totals[name][2] += span.get("minimum_tensor_bytes", 0)
    result = []
    for name, (seconds, count, tensor_bytes) in sorted(
        totals.items(), key=lambda p: -p[1][0]
    ):
        row = dict(
            name=name,
            seconds=seconds,
            count=count,
            minimum_tensor_gbps=tensor_bytes / max(seconds, 1e-12) / 1e9,
            minimum_tensor_bandwidth_fraction=tensor_bytes
            / max(seconds, 1e-12)
            / bandwidth_peak,
        )
        attention = re.fullmatch(r"attention:M(\d+):K(\d+):H(\d+):D(\d+)", name)
        if attention:
            m, k, h, d = map(int, attention.groups())
            flops = count * 4 * h * d * (m * (k - m) + m * (m + 1) / 2)
            row.update(
                tflops=flops / seconds / 1e12,
                compute_peak_fraction=flops / seconds / compute_peak,
                arithmetic_floor_s=flops / compute_peak,
            )
        match = re.fullmatch(r"gemm:M(\d+):K(\d+):N(\d+):b(\d+)", name)
        if match:
            m, k, n, bits = map(int, match.groups())
            flops = count * 2 * m * k * n
            # Minimum tensor traffic, NOT measured DRAM transactions or tile rereads.
            tensor_bytes = count * (n * k * bits / 8 + n * k / 64 * 4 + 2 * m * (k + n))
            row.update(
                tflops=flops / seconds / 1e12,
                compute_peak_fraction=flops / seconds / compute_peak,
                minimum_tensor_gbps=tensor_bytes / seconds / 1e9,
                minimum_tensor_bandwidth_fraction=tensor_bytes
                / seconds
                / bandwidth_peak,
                arithmetic_floor_s=flops / compute_peak,
            )
        result.append(row)
    return dict(
        wall_s=wall,
        gpu_busy_union_s=busy,
        cb_idle_gaps_s=gaps,
        wall_minus_gpu_busy_s=wall - busy,
        spans=result,
        warning="Synchronized diagnostics: do not scale into an uninstrumented wall-time claim.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--log")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--out")
    args = parser.parse_args()
    cells = validate_cold(
        [json.loads(x) for x in Path(args.rows).read_text().splitlines()],
        expected_tokens=args.tokens,
    )
    result = dict(cells=cells, complete=True)
    if args.log:
        records = [
            json.loads(line.split("COLDTTFT_PROFILE ", 1)[1])
            for line in Path(args.log).read_text().splitlines()
            if "COLDTTFT_PROFILE " in line
        ]
        result["profile"] = summarize(records)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
