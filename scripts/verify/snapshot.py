"""Cross-engine snapshot verdicts through yv, with one resident engine at a time."""

from __future__ import annotations

import sys
from pathlib import Path

from .execute import Cell


def stage_snapshot(ctx):
    from .stages import StageResult, _finish

    harness = ctx.cand.path / "scripts/research"
    sys.path.insert(0, str(harness))
    import bench_snapshot as bs

    engines = [
        e
        for e in ctx.env.get("SNAPSHOT_ENGINES", ",".join(bs.ENGINE_ORDER)).split(",")
        if e
    ]
    unknown = set(engines) - set(bs.ENGINE_ORDER)
    if unknown:
        raise ValueError(f"unknown snapshot engines: {sorted(unknown)}")
    outdir = ctx.run.path / "snapshot"
    trees = {"yunshu-new": str(ctx.base.path / "python")}
    jobs = bs.plan_pilots(engines, outdir, trees)
    jobs += bs.plan_cells(engines, 3, outdir, trees)
    failed_engines = set()
    pilot_failed = set()
    reasons, evidence = [], {}
    for job in jobs:
        job.env["TFB_OUT"] = str(ctx.run.path / "work")
        if job.engine == "yunshu-new":
            job.env["TFB_EXPECT_YUNSHU_SHA"] = ctx.base.commit
        if job.engine in failed_engines:
            continue

        def validate(path, job=job):
            problems = bs.validate_rows(job, bs.read_rows(path))
            return not problems, "; ".join(problems)

        argv = [
            "env",
            *(f"{k}={v}" for k, v in job.env.items()),
            *bs.tfbench_argv(job, Path("{out}")),
        ]
        cell = Cell(
            "snapshot",
            job.name,
            argv,
            mem_gb=job.mem_gb,
            timeout_min=job.timeout_min,
            stall_min=job.stall_min,
            quiet=job.stage != "pilot",
            device="m5",
            validate=validate,
            retries=1 if job.stage != "pilot" else 0,
        )
        result = ctx.exe.run_cells([cell])[job.name]
        if not result.ok:
            reasons.append(f"{job.name}: {result.reason}")
            # First bad request fails fast for this engine, while keeping
            # other engine coverage and clearly reporting missing cells.
            failed_engines.add(job.engine)
            if job.stage == "pilot":
                pilot_failed.add(job.engine)
        else:
            evidence[job.name] = str(result.evidence)
    numbers = {
        "release_sha": ctx.base.key,
        "harness_sha": ctx.cand.key,
        "engines": engines,
        "failed_engines": sorted(failed_engines),
        "evidence": evidence,
        "complete_jobs": len(evidence),
        "planned_jobs": len(jobs),
    }
    if ctx.suite.get("snapshot_agents", True):
        import snapshot_agent

        agent_results = {}
        tasks = bs.agent_tasks()
        for engine in engines:
            if engine in pilot_failed:
                continue
            for task in tasks:
                key = f"agent-{engine}-{task}"

                def agent_validate(path, task=task):
                    return snapshot_agent.validate(bs.read_rows(path), task)

                env = bs.job_env(engine, trees)
                env["TFB_OUT"] = str(ctx.run.path / "work")
                if engine == "yunshu-new":
                    env["TFB_EXPECT_YUNSHU_SHA"] = ctx.base.commit
                cell = Cell(
                    "snapshot",
                    key,
                    [
                        "env",
                        *(f"{k}={v}" for k, v in env.items()),
                        ctx.py,
                        str(harness / "snapshot_agent.py"),
                        "--engine",
                        engine,
                        "--task",
                        task,
                        "--out",
                        "{out}",
                    ],
                    mem_gb=60,
                    timeout_min=28,
                    stall_min=24,
                    quiet=True,
                    device="m5",
                    validate=agent_validate,
                    retries=1,
                )
                result = ctx.exe.run_cells([cell])[key]
                if not result.ok:
                    reasons.append(f"{key}: {result.reason}")
                    break
                rows = bs.read_rows(result.evidence)
                row = next(r for r in rows if r.get("type") == "run")
                agent_results[key] = {"evidence": str(result.evidence), **row}
                if any(
                    row.get(k, 0)
                    for k in (
                        "api_errors",
                        "malformed_tool_calls",
                        "leaked_tool_markup",
                    )
                ):
                    reasons.append(
                        f"{key}: agent API/tool contract failure; remaining tasks unknown"
                    )
                    break
        numbers["agent_tasks"] = tasks
        numbers["agent_results"] = agent_results
    tables = bs.render_markdown(bs.collect(outdir, engines, ctx.run.path), engines)
    (ctx.run.path / "snapshot_tables.md").write_text(tables)
    return _finish(ctx, StageResult("snapshot", not reasons, reasons, numbers))
