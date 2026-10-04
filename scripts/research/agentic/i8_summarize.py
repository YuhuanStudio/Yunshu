"""Fail closed on missing session runs; report paired I8 latency / output identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def percentiles(rows: list[dict]) -> dict:
    return dict(
        zip(
            ("p50", "p90"),
            np.percentile([r["ttft_s"] * 1000 for r in rows], [50, 90]).tolist(),
            strict=True,
        )
    )


def summarize(root: Path, first_run: int = 1) -> dict:
    records = {}
    titles = {}
    source = None
    for mode in ("0", "1"):
        records[mode], titles[mode] = [], []
        for run in range(first_run, first_run + 3):
            path = root / f"{mode}-{run}.jsonl"
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            mains = [r for r in rows if r["kind"] == "main"]
            auxiliary = [r for r in rows if r["kind"] == "title"]
            if (
                len(rows) != 8
                or len(mains) != 6
                or len(auxiliary) != 1
                or rows[-1]["kind"] != "complete"
            ):
                raise ValueError(f"incomplete session replay: {path}")
            receipt = rows[-1]
            if receipt.get("engaged_mode") != "mtp" or not receipt.get("source_sha256"):
                raise ValueError(f"missing MTP / source receipt: {path}")
            if source is None:
                source = receipt["source_sha256"]
            elif source != receipt["source_sha256"]:
                raise ValueError(f"different source versions: {path}")
            if any(not r["stream_done"] for r in [*mains, *auxiliary]):
                raise ValueError(f"incomplete stream: {path}")
            if mains[0].get("cached_tokens") != 0:
                raise ValueError(f"first agent turn was not cold: {path}")
            records[mode].append(mains)
            titles[mode].extend(auxiliary)
    result = {
        "source_sha256": source,
        "run_numbers": list(range(first_run, first_run + 3)),
    }
    for mode, runs in records.items():
        result[mode] = dict(
            runs=3,
            main_requests=18,
            title_requests=3,
            main_ttft_ms=percentiles([r for run in runs for r in run]),
            per_run_main_ttft_ms=[percentiles(run) for run in runs],
            cold_main_ttft_ms=[run[0]["ttft_s"] * 1000 for run in runs],
            warm_main_ttft_ms=percentiles([r for run in runs for r in run[1:]]),
        )
    pairs = [
        (b, a)
        for br, ar in zip(records["0"], records["1"], strict=True)
        for b, a in zip(br, ar, strict=True)
    ]
    title_pairs = list(zip(titles["0"], titles["1"], strict=True))
    for b, a in [*pairs, *title_pairs]:
        if (b["body"], b["seed"]) != (a["body"], a["seed"]):
            raise ValueError("unpaired replay bodies / seeds")
    if any(r.get("cached_tokens") != 0 for r in titles["1"]):
        raise ValueError("auxiliary APC exclusion not observed")
    result["after_title_cached_tokens"] = [r["cached_tokens"] for r in titles["1"]]
    result["identity"] = dict(
        main_equal=sum(b["output_sha256"] == a["output_sha256"] for b, a in pairs),
        title_equal=sum(
            b["output_sha256"] == a["output_sha256"] for b, a in title_pairs
        ),
        main_pairs=len(pairs),
        title_pairs=len(title_pairs),
    )
    result["main_ttft_reduction_percent"] = {
        q: 100 * (1 - result["1"]["main_ttft_ms"][q] / result["0"]["main_ttft_ms"][q])
        for q in ("p50", "p90")
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--first-run", type=int, default=1)
    args = parser.parse_args()
    result = summarize(args.root, args.first_run)
    output = json.dumps(result, indent=2) + "\n"
    (args.root / "summary.json").write_text(output)
    print(output)
    identity = result["identity"]
    if (
        identity["main_equal"] != identity["main_pairs"]
        or identity["title_equal"] != identity["title_pairs"]
    ):
        raise SystemExit("output identity failed")


if __name__ == "__main__":
    main()
