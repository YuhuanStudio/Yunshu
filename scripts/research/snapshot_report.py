"""CPU-only reporting of promoted yv cross-engine evidence, including failures."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import bench_snapshot as bs


def recall_table(cells, engines):
    output = [
        "## Long QA gap to the best (percentage points)",
        "",
        "| ctx | " + " | ".join(engines) + " |",
        "|---|" + "---|" * len(engines),
    ]
    for ctx in (32768, 65536, 131072):
        values = [
            cells.get((engine, "needle", ctx, "prose", "correct"), [])
            for engine in engines
        ]
        complete = [sum(v) / len(v) for v in values if len(v) == 10]
        best = max(complete) if complete else None
        entries = [
            f"{int(sum(v))}/10; gap {100 * (best - sum(v) / 10):.1f} pp"
            if len(v) == 10
            else "unknown"
            for v in values
        ]
        output.append(f"| {ctx // 1024}K | " + " | ".join(entries) + " |")
    output += [
        "",
        "Recall gaps may reflect different target quantizations, long-context arithmetic, prefix state handling or retrieval limits.",
        "",
    ]
    return "\n".join(output)


def agent_table(rows, engines, expected=20):
    grouped = {engine: [] for engine in engines}
    for row in rows:
        if row.get("type") == "run" and row.get("engine") in grouped:
            grouped[row["engine"]].append(row)
    rates = {
        e: sum(bool(r.get("passed")) for r in v) / len(v)
        for e, v in grouped.items()
        if len(v) == expected
    }
    best = max(rates.values()) if rates else None
    output = [
        "## Agentbench (same 20 opencode tasks, greedy)",
        "",
        "| engine | graded / planned | pass / graded | gap to best | API / malformed / markup | wall median (s) |",
        "|---|---|---|---|---|---|",
    ]
    for engine, values in grouped.items():
        passed = sum(bool(r.get("passed")) for r in values)
        gap = f"{100 * (best - rates[engine]):.1f} pp" if engine in rates else "unknown"
        errors = (
            "/".join(
                str(sum(r.get(k, 0) for r in values))
                for k in ("api_errors", "malformed_tool_calls", "leaked_tool_markup")
            )
            if values
            else "unknown"
        )
        wall = [r["wall_s"] for r in values if isinstance(r.get("wall_s"), int | float)]
        output.append(
            f"| {engine} | {len(values)}/{expected} | {passed}/{len(values)} | {gap} | {errors} | "
            + (f"{statistics.median(wall):.1f}" if wall else "unknown")
            + " |"
        )
    output += [
        "",
        "Agent pass-rate gaps may come from tool/API contracts, prefix reuse, context handling and the model's coding answers.",
        "A partial task set is not ranked; different target quantizations remain a quality confound.",
        "",
    ]
    return "\n".join(output)


def summarize(run, engines=None):
    engines = engines or list(bs.ENGINE_ORDER)
    state_path = run / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    events = bs.read_rows(run / "snapshot.jsonl")
    done = [r for r in events if r.get("ev") == "cell_done"]
    latest = {r["cell"]: r for r in done}
    failures = [r for r in latest.values() if not r.get("ok")]
    agents, identity = [], []
    for event in latest.values():
        if not event.get("ok") or event.get("rc") != 0 or event.get("contended"):
            continue
        path = run / "cells" / f"snapshot.{event['cell']}.jsonl"
        rows = bs.read_rows(path)
        if event["cell"].startswith("agent-"):
            agents += [r for r in rows if r.get("type") == "run"]
        cold = {
            (r.get("ctx"), r.get("kind")): r
            for r in rows
            if r.get("part") == "decode" and r.get("phase") == "cold"
        }
        for row in rows:
            if row.get("part") == "decode" and row.get("phase") == "warm":
                base = cold.get((row.get("ctx"), row.get("kind")))
                if base and base.get("sha") != row.get("sha"):
                    identity.append(
                        {
                            "cell": event["cell"],
                            "ctx": row.get("ctx"),
                            "kind": row.get("kind"),
                            "cold": base.get("sha"),
                            "warm": row.get("sha"),
                        }
                    )
    cells = bs.collect(run / "snapshot", engines, run)
    tables = (
        bs.render_markdown(cells, engines)
        + "\n"
        + recall_table(cells, engines)
        + "\n"
        + agent_table(agents, engines)
    )
    tables += "\n## Evidence and failures\n\n"
    tables += f"Release `{state.get('base', {}).get('commit', 'unknown')}`; harness `{state.get('cand', {}).get('commit', 'unknown')}`.\n\n"
    tables += "| cell | job | rc | status / reason |\n|---|---|---|---|\n"
    for event in done:
        reason = (
            str(event.get("reason") or "valid").replace("|", "\\|").replace("\n", " ")
        )
        tables += (
            f"| {event['cell']} | {event.get('job')} | {event.get('rc')} | {reason} |\n"
        )
    tables += f"\nCold/warm output digest mismatches: **{len(identity)}** (diagnostic, not cross-engine bit identity).\n"
    return {
        "complete": state.get("status") == "finished",
        "status": state.get("status", "unknown"),
        "jobs": done,
        "failures": failures,
        "identity_mismatches": identity,
        "agent_runs": agents,
        "markdown": tables,
    }


def combined(sources):
    """One explicit evidence source per engine; never pool different runs."""
    releases, cells, agent_runs, jobs, failures, mismatches, provenance = (
        set(),
        {},
        [],
        [],
        [],
        [],
        {},
    )
    complete = True
    for engine, run in sources.items():
        state = json.loads((run / "state.json").read_text())
        releases.add(state["base"]["commit"])
        result = summarize(run, [engine])
        complete &= result["complete"]
        cells.update(bs.collect(run / "snapshot", [engine], run))
        agent_runs += [r for r in result["agent_runs"] if r.get("engine") == engine]
        selected = [
            r
            for r in result["jobs"]
            if r["cell"].startswith(engine + "-")
            or r["cell"].startswith("agent-" + engine + "-")
        ]
        jobs += selected
        failures += [r for r in selected if not r.get("ok")]
        mismatches += [
            r
            for r in result["identity_mismatches"]
            if r["cell"].startswith(engine + "-")
        ]
        provenance[engine] = {
            "run": str(run),
            "release": state["base"]["commit"],
            "harness": state["cand"]["commit"],
        }
    if len(releases) != 1:
        raise ValueError("cannot combine different released engine commits")
    engines = list(sources)
    markdown = (
        bs.render_markdown(cells, engines)
        + "\n"
        + recall_table(cells, engines)
        + "\n"
        + agent_table(agent_runs, engines)
    )
    markdown += "\n## Provenance\n\n| engine | release | harness |\n|---|---|---|\n"
    for engine, source in provenance.items():
        markdown += f"| {engine} | {source['release']} | {source['harness']} |\n"
    markdown += "\n## Recorded failures (including retried attempts)\n\n| cell | job | rc | reason |\n|---|---|---|---|\n"
    for event in failures:
        reason = str(event.get("reason", "")).replace("|", "\\|").replace("\n", " ")
        markdown += (
            f"| {event['cell']} | {event.get('job')} | {event.get('rc')} | {reason} |\n"
        )
    return {
        "complete": bool(complete),
        "release": next(iter(releases)),
        "provenance": provenance,
        "jobs": jobs,
        "failures": failures,
        "identity_mismatches": mismatches,
        "agent_runs": agent_runs,
        "markdown": markdown,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="ENGINE=RUN, one canonical run per engine",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.source:
        sources = {
            engine: Path(run)
            for engine, run in (entry.split("=", 1) for entry in args.source)
        }
        if len(sources) != len(args.source):
            parser.error("duplicate engine evidence sources")
        report = combined(sources)
    elif args.run:
        report = summarize(args.run)
    else:
        parser.error("--run or --source is required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.pop("markdown"))
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("jobs", "agent_runs", "identity_mismatches", "failures")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
