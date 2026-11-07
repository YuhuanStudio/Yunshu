"""Concurrent (conc / mixed) cells launch each engine configured for N clients, the way a user serving N clients
would per the engine's own docs (lead decision 2026-11-04): TensorFold --parallel N with --context sized to the cell,
mlx-serve --max-concurrent N, Kapitsa --slots N, llama.cpp -np N with the context per slot = the cell, oMLX
--max-concurrent-requests >= N. The header records the launch; the audit fails a lower N; an engine that dies at N is
'unsupported at N (measured: ...)'."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import bench_engines as be  # noqa: E402
import fullcmp_audit as fa  # noqa: E402
import fullcmp_run as fr  # noqa: E402
import tfbench as tb  # noqa: E402


def _launch(engine, parallel, ctx):
    return be.build_launch(
        engine, 18990, "/tmp/h", {}, ctx_tokens=ctx, parallel=parallel
    )


def _val(cmd, flag):
    return list(cmd)[list(cmd).index(flag) + 1]


def test_cell_context_is_the_longest_request_plus_output_plus_margin_rounded_up_to_1024():
    assert (
        fr.cell_context({"ctx_tokens": 1024, "out_len": 512}) == 3072
    )  # 1024 + 512 + 1024 = 2560, rounded up to 1024
    assert fr.cell_context({"ctx_tokens": 8192, "out_len": 512}) == 10240
    assert fr.cell_context({"ctx_tokens": 131072, "out_len": 512}) % 1024 == 0
    assert (
        fr.cell_context({"ctx_tokens": 131072, "out_len": 512}) >= 131072 + 512 + 1024
    )


def _row(engine, conc, turns="single", ctx=1024):
    return {
        "engine": engine,
        "conc": conc,
        "turns": turns,
        "ctx_tokens": ctx,
        "ctx": "1K",
        "out_len": 512,
        "prompt_kind": "code",
        "sweep": "conc",
        "batch": "B2",
        "model": "q27-4bit",
        "id": "x",
        "est_gpu_min": 1,
        "reps": 1,
        "phases": ("conc",),
        "sampling": "greedy",
        "table": "decode",
        "thinking": "off",
        "media": "none",
    }


def test_harness_part_passes_parallel_for_every_concurrent_cell_and_the_cell_context_for_the_per_stream_engines():
    part, args, _ = fr.harness_part(_row("tf-1.2.1", 8))
    assert part == "conc" and args[args.index("--parallel") + 1] == "8"
    assert args[args.index("--ctx-tokens") + 1] == "3072"
    _, args, _ = fr.harness_part(_row("kapitsa-0.2.26", 4, turns="mixed", ctx=8192))
    assert (
        args[args.index("--parallel") + 1] == "4"
        and args[args.index("--ctx-tokens") + 1] == "10240"
    )
    # an engine without a per-stream context keeps its launch context: only --parallel
    _, args, _ = fr.harness_part(_row("yunshu", 8))
    assert "--parallel" in args and "--ctx-tokens" not in args
    # a conc-1 decode cell is unchanged
    part, args, _ = fr.harness_part(_row("tf-1.2.1", 1))
    assert part == "decode" and "--parallel" not in args


def test_each_engine_launches_for_n_clients():
    tf = _launch("tf-1.2.1", 8, 2560).cmd
    assert _val(tf, "--parallel") == "8" and _val(tf, "--context") == "2560"
    ms = _launch("mlx-serve-26.11.1", 8, 2560).cmd
    assert _val(ms, "--max-concurrent") == "8" and _val(ms, "--ctx-size") == "2560"
    kp = _launch("kapitsa-0.2.26", 8, 2560).cmd
    assert (
        _val(kp, "--slots") == "8"
        and _val(kp, "--context") == "2560"
        and "--private" in kp
    )
    ll = _launch("llama-cpp-b12061", 8, 2560).cmd
    assert _val(ll, "-np") == "8" and _val(ll, "-c") == str(
        2560 * 8
    )  # the context per slot = the cell
    assert (
        _val(_launch("omlx-0.7.3", 16, 2560).cmd, "--max-concurrent-requests") == "16"
    )
    assert (
        _val(_launch("omlx-0.7.3", 4, 2560).cmd, "--max-concurrent-requests") == "8"
    )  # the default 8 stays


def test_a_decode_cell_launch_is_unchanged_except_kapitsa_private():
    assert "--max-concurrent" not in _launch("mlx-serve-26.11.1", 1, 140000).cmd
    kp = _launch("kapitsa-0.2.26", 1, 140000).cmd
    assert "--slots" not in kp and "--context" not in kp and "--private" in kp
    ll = _launch("llama-cpp-b12061", 1, 140000).cmd
    assert "-np" not in ll and _val(ll, "-c") == "140000"
    assert _val(_launch("tf-1.2.1", 1, 140000).cmd, "--parallel") == "1"


def _cf(engine, n, cmd, log_text=None, tmp=None):
    cell = {"cell": f"B2.conc.{engine}", "engine": engine, "group": "g", "conc": n}
    path = Path("x.jsonl")
    if log_text is not None:
        path = tmp / "c-r0.jsonl"
        (tmp / "c-r0.server.log").write_text(log_text)
    return fa.CellFile(
        cell, 0, path, {"part": "session", "cmd": cmd, "home": "/h", "job_id": "j"}, []
    )


def _concurrency_violations(*cfs):
    res = fa.GuardResult("fairness_controls")
    fa.check_launch_concurrency(
        fa.Run(manifest={}, files=list(cfs), missing=[], extra=[], parse_errors=[]), res
    )
    return res.violations


def test_audit_fails_a_concurrent_cell_launched_below_n_and_passes_the_configured_one():
    assert (
        _concurrency_violations(_cf("tf-1.2.1", 8, ["tf", "serve", "--parallel", "8"]))
        == []
    )
    assert (
        _concurrency_violations(
            _cf("kapitsa-0.2.26", 8, ["k", "serve", "--slots", "8"])
        )
        == []
    )
    low = _concurrency_violations(
        _cf("tf-1.2.1", 8, ["tf", "serve", "--parallel", "1"])
    )
    assert low and "needs >= 8" in low[0]
    assert _concurrency_violations(
        _cf("mlx-serve-26.11.1", 4, ["ms", "serve"])
    )  # no flag at all: default 1
    assert (
        _concurrency_violations(_cf("yunshu", 8, ["y"])) == []
    )  # no such flag: admits concurrently
    assert (
        _concurrency_violations(_cf("tf-1.2.1", 1, ["tf", "serve"])) == []
    )  # a decode cell


def test_audit_fails_a_tensorfold_server_that_started_fewer_streams_than_n(tmp_path):
    good = _cf(
        "tf-1.2.1",
        8,
        ["tf", "--parallel", "8"],
        "info: Qwen3.8-27B: 8 streams of 2560 tokens\n",
        tmp_path,
    )
    assert _concurrency_violations(good) == []
    bad = _cf(
        "tf-1.2.1",
        8,
        ["tf", "--parallel", "8"],
        "info: Qwen3.8-27B: 1 streams of 2560 tokens\n",
        tmp_path,
    )
    assert any("started 1 streams for N=8" in v for v in _concurrency_violations(bad))


def test_a_stub_run_is_exempt():
    res = fa.GuardResult("x")
    run = fa.Run(
        manifest={"stub_engine": True},
        files=[_cf("tf-1.2.1", 8, ["stub"])],
        missing=[],
        extra=[],
        parse_errors=[],
    )
    fa.check_launch_concurrency(run, res)
    assert res.violations == []


def test_an_engine_that_dies_at_n_is_unsupported_at_n_and_other_failures_still_raise(
    tmp_path,
):
    out_path = tmp_path / "cell-r0.jsonl"
    a = types.SimpleNamespace(parallel=8, part="mixed", out=str(out_path), rep=0)
    dead = types.SimpleNamespace(proc=types.SimpleNamespace(poll=lambda: -9))
    alive = types.SimpleNamespace(proc=types.SimpleNamespace(poll=lambda: None))

    def boom(s, out, a):
        raise ConnectionError("connection reset")

    with out_path.open("w") as out:
        tb.run_part_or_unsupported(boom, dead, out, a)
    rows = [json.loads(x) for x in out_path.read_text().splitlines()]
    un = [r for r in rows if r["part"] == "unsupported"]
    assert (
        len(un) == 1
        and "unsupported at N=8 (measured: engine exited rc=-9" in un[0]["reason"]
    )
    assert tb.cell_unsupported(str(out_path)) is True
    with out_path.open("w") as out, pytest.raises(ConnectionError):
        tb.run_part_or_unsupported(
            boom, alive, out, a
        )  # the engine is alive: a real failure
    a1 = types.SimpleNamespace(parallel=1, part="mixed", out=str(out_path), rep=0)
    with out_path.open("w") as out, pytest.raises(ConnectionError):
        tb.run_part_or_unsupported(boom, dead, out, a1)  # N = 1: never converted


def test_the_c8_validation_manifest_has_conc_c8_and_mixed_c4_for_each_engine_with_a_parallel_flag():
    import fullcmp_manifest as fm

    rows = fm.wire_1108c8_rows()
    by = {(r["engine"], r["sweep"]): r for r in rows}
    for eng in fm.WIRE_1108C8_ENGINES:
        assert by[(eng, "conc")]["conc"] == 8 and by[(eng, "conc")]["ctx"] == "1K", eng
        assert by[(eng, "mixed")]["conc"] == 4 and by[(eng, "mixed")]["ctx"] == "8K", (
            eng
        )
        assert by[(eng, "conc")]["status"] == "unverified"
    # every engine of the smoke has a recorded parallel flag the audit checks
    assert all(e in be.PARALLEL_FLAG for e in fm.WIRE_1108C8_ENGINES)
    # the harness turns each row into a concurrent tfbench command with N and the cell context
    for r in rows:
        part, args, _ = fr.harness_part(r)
        assert part in ("conc", "mixed") and args[args.index("--parallel") + 1] == str(
            r["conc"]
        )
