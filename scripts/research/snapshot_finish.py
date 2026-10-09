"""CPU-only completion collector; never submits GPU work or changes engine trees."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import bench_snapshot as bs
import snapshot_report


def all_finished(sources):
    return all(
        (run / "state.json").exists()
        and json.loads((run / "state.json").read_text()).get("status") == "finished"
        for run in sources.values()
    )


def write_results(
    root,
    sources,
    last_path=Path("/Volumes/P5Plus/yunshu-build/codex/snapshot014_last.md"),
):
    result = snapshot_report.combined(sources)
    if not result["complete"]:
        raise RuntimeError("measurement controllers have not finished")
    private = root / "docs/research/snapshot014"
    private.mkdir(parents=True, exist_ok=True)
    (private / "final_tables.md").write_text(result["markdown"])
    payload = {k: v for k, v in result.items() if k != "markdown"}
    (private / "final_evidence.json").write_text(json.dumps(payload, indent=2) + "\n")
    benchmarks = root / "docs/BENCHMARKS.md"
    intro = benchmarks.read_text().split("# Metric tables", 1)[0]
    intro = intro.replace(
        "Snapshot in progress: **unmeasured/rejected cells are unknown**.",
        "Snapshot controllers finished: **unmeasured/rejected cells are unknown**.",
    )
    benchmarks.write_text(
        intro + result["markdown"].replace("# Snapshot tables", "# Metric tables", 1)
    )
    jobs = result["jobs"]
    # Include every attempt, even an earlier startup failure superseded by a
    # canonical per-engine retry. These are diagnostics, never timing samples.
    history = {}
    initial = private / "initial_jobs.json"
    if initial.exists():
        for event in json.loads(initial.read_text()):
            history[event["job"]] = event
    for run in set(sources.values()):
        for event in bs.read_rows(run / "snapshot.jsonl"):
            if event.get("ev") == "cell_done":
                history[event["job"]] = event
    history_table = "\n## 全部 job attempts（含被修正版取代的啟動失敗）\n\n| job | rc | 結果／原因 |\n|---|---|---|\n"
    for event in history.values():
        reason = (
            str(event.get("reason") or "complete / valid")
            .replace("|", "\\|")
            .replace("\n", " ")
        )
        history_table += f"| {event['job']} | {event.get('rc')} | {reason} |\n"
    ids = [r.get("job", "unknown") for r in jobs]
    valid = sum(bool(r.get("ok")) for r in jobs)
    failures = result["failures"]
    trend = (
        "\n\n## snapshot014 — v0.1.4 cross-engine snapshot (M5 Max 128 GB)\n\n"
        f"Released engine: `{result['release']}`. Complete reference inputs at 1K/8K/32K/64K/128K; greedy, 2048-token replies.\n\n"
        f"Recorded job attempts: {len(jobs)}; valid attempts: {valid}; failed/rejected attempts: {len(failures)}.\n\n"
        "No engine optimization/default change is claimed. Different Splash/GGUF target quantizations remain confounds.\n"
        "Tables, per-metric gaps, hypotheses, partial agent coverage and failures: [BENCHMARKS](../BENCHMARKS.md).\n\n"
        "Harness provenance: "
        + ", ".join(f"{e} `{p['harness']}`" for e, p in result["provenance"].items())
        + ".\n\n"
        "Job IDs: " + ", ".join(ids) + ".\n"
    )
    path = root / "docs/reports/PERF_TREND.md"
    if "## snapshot014 — v0.1.4 cross-engine snapshot" not in path.read_text():
        with path.open("a") as stream:
            stream.write(trend)
    release = (
        "Yunshu v0.1.4 improves long-context scheduling and prefix reuse on a single Apple Silicon machine, "
        "extends the lossless DFlash fast path to long replies, and fixes forced-tool calls and Responses compaction. "
        "The released defaults were compared with TensorFold, mlx-lm, oMLX, Splash, llama.cpp and MTPLX on one M5 Max "
        "using a shared greedy corpus and exactly 131072-token reference inputs at 128K. BENCHMARKS.md reports each "
        "metric, gap to the measured best and a hypothesis; failures and unmeasured results are explicit, and different "
        "target quantizations are identified rather than treated as same-checkpoint equivalence."
    )
    report = (
        "# snapshot014 測量流程完成\n\n"
        f"Release `{result['release']}`；已記錄 {len(jobs)} 個 job attempts，其中 {valid} valid、{len(failures)} failed/rejected。\n"
        "控制流程完成不表示每個引擎／cell 都成功；unknown 不當作通過。所有數字是 M5。\n\n"
        "## 改動與 CPU 驗證\n\n"
        "bench014 修正完整 131072-token 輸入、隔離啟動、bind retry／listener ownership、yv snapshot 與 per-engine provenance。\n"
        "準備階段完整 unit：9774 passed、6 skipped；ruff check／format 與 mypy gate 通過。最終 main 同步／交付 SHA 檢查仍須由 worker 或 lead 核對。\n\n"
        "## 實測、差距與失敗\n\n"
        + result["markdown"]
        + "\n\n"
        + history_table
        + "\n## 尚待完成\n\n"
        "未知／失敗 cells 詳見上表。需核對最終 main 同步、final-SHA yv 與完整 unit 後才可宣告 READY TO MERGE；本 collector 不 merge、push 或 stash。\n\n"
        "## own ideas\n\n"
        "把 prefix reuse 拆成可量化的 cache debt：相同 prefix 下，對照 full-hit、尾端小幅修改與中段分支，以 cached coverage 分解搜尋、hybrid-state restore、重新 prefill 與 first-token dispatch。\n"
        "再把 agent loop 的固定成本按每次工具回合累積，辨認單次只有幾十毫秒、總流程卻明顯的瓶頸。這些是待驗證方向，不宣稱已取得改善。\n\n"
        "## GitHub v0.1.4 release-notes paragraph\n\n" + release + "\n"
    )
    last_path.write_text(report)
    (private / "FINAL_REPORT.md").write_text(report)
    return {
        "complete": True,
        "jobs": len(jobs),
        "valid": valid,
        "failures": len(failures),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)
    sources = {e: Path(p) for e, p in json.loads(args.sources.read_text()).items()}
    while not all_finished(sources):
        if not args.wait:
            print(json.dumps({"complete": False, "status": "waiting"}))
            return 2
        time.sleep(60)
    print(json.dumps(write_results(args.root, sources)))
    # Read queue outcomes/log tails for every canonical job without touching servers.
    result = snapshot_report.combined(sources)
    for event in result["jobs"]:
        job = event.get("job", "")
        if not job.startswith("cache:"):
            subprocess.run(
                [bs.GPUQ, "log", job], stdout=subprocess.DEVNULL, check=False
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
