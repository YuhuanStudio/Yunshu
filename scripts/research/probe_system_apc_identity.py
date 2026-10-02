"""Separate hybrid prefix restore identity from decode batching (gpuq only)."""

import argparse
import json
import time
from pathlib import Path

import bench_prefix_singleflight as b
import tfbench as t
from cache_probe_source import freeze


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    source = freeze(a.out, t)
    prefix = "System APC identity.\n" + t.load_prompt("prose-8192")
    results = {}
    logs = []
    for arm in ("cold", "prime"):
        s = t.Srv(
            "yunshu",
            {"YUNSHU_VLM_APC_DISK": "0", "YUNSHU_VLM_DRAFT": t.D},
            "cachesf-sysid-" + arm + "-" + str(time.time_ns()),
        )
        try:
            b.ask(s, "Warmup.", 0, False)
            if arm == "prime":
                results["prime"] = b.ask(s, prefix, 1, False)
            results[arm] = b.ask(s, prefix, 0, False)
            if arm == "cold":
                results["exact"] = b.ask(s, prefix, 0, False)
        finally:
            s.kill()
        logs.append(str(s.log))
    reference = results["cold"]
    checks = {}
    for arm in ("exact", "prime"):
        r = results[arm]
        checks[arm] = {
            "cached": r["cached"],
            "tokens_equal": r["tokens"] == reference["tokens"],
            "logprobs_equal": r["lps"] == reference["lps"],
            "max_abs_delta": max(
                abs(x - y) for x, y in zip(r["lps"], reference["lps"], strict=True)
            ),
        }
    ok = all(c["tokens_equal"] and c["logprobs_equal"] for c in checks.values())
    record = {
        "complete": True,
        "ok": ok,
        "checks": checks,
        "requests": results,
        "source": source,
        "server_logs": logs,
    }
    Path(a.out).write_text(json.dumps(record) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "requests"}), flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
