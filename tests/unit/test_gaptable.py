"""CPU fixtures for the public cross-engine harness (no MLX)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _private_environ(monkeypatch):
    """gaptable.run() sets TMPDIR / TFB_* for its own process; inside the test process those would leak into later
    tests on the same xdist worker (a deleted TMPDIR broke MLX and the zsh sandbox tests in ci-local)."""
    import os

    monkeypatch.setattr(os, "environ", os.environ.copy())


ROOT = Path(__file__).parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / f"scripts/research/{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gt = load("gaptable")
tb = load("tfbench")


def test_tf_slots_do_not_reserve_eight_long_streams(tmp_path):
    be = tb.bench_engines
    launch = be.build_launch("tf-native", 18996, tmp_path, {}, 36864, 1)
    assert launch.cmd[launch.cmd.index("--parallel") + 1] == "1"
    assert launch.cmd[launch.cmd.index("--context") + 1] == "36864"


def test_tf_standard_usage_and_unknown_are_distinct():
    assert (
        tb.cached_tokens(
            {"prompt_tokens": 32768, "prompt_tokens_details": {"cached_tokens": 32761}}
        )
        == 32761
    )
    assert (
        tb.cached_tokens(
            {"prompt_tokens": 32768, "prompt_tokens_details": {"cached_tokens": 0}}
        )
        == 0
    )
    assert tb.cached_tokens({"prompt_tokens": 32768}) is None
    for value in (-1, 32769, True, "0"):
        with pytest.raises(ValueError):
            tb.cached_tokens(
                {
                    "prompt_tokens": 32768,
                    "prompt_tokens_details": {"cached_tokens": value},
                }
            )


def test_parser_keeps_tf_cached_usage(monkeypatch):
    import io

    frames = [
        dict(choices=[dict(delta={"content": "ok"}, finish_reason="length")]),
        dict(
            choices=[],
            usage=dict(
                prompt_tokens=32768,
                completion_tokens=512,
                prompt_tokens_details=dict(cached_tokens=32761),
            ),
        ),
    ]
    payload = (
        b"".join(b"data: " + json.dumps(f).encode() + b"\n\n" for f in frames)
        + b"data: [DONE]\n\n"
    )
    monkeypatch.setattr(
        tb.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload)
    )
    assert tb.send("http://fixture", {})["cached"] == 32761


def test_interleave_three_reps_and_missing_rows():
    plan = gt.schedule(["yunshu-main", "tf-native", "splash-noane"])
    assert len(plan) == 7 * 3 * 3
    assert [r[1] for r in plan[:6]] == [
        "yunshu-main",
        "tf-native",
        "splash-noane",
        "tf-native",
        "splash-noane",
        "yunshu-main",
    ]
    assert ("conc", 1024, "mixed") in gt.cells()
    assert all(c[1] <= 32768 for c in gt.cells(True))


def test_warm_cache_miss_refuses_a_table_cell(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(dict(part="decode", phase="warm", cached=0))
        + '\n{"part":"part_done"}\n'
    )
    with pytest.raises(RuntimeError, match="invalid warm"):
        gt.validate(path, "decode", 32768)


def test_concurrent_prompt_exact_size_and_salt(monkeypatch):
    import snapshot_prompts

    seen = []
    monkeypatch.setattr(tb, "load_prompt", lambda n: "source")
    monkeypatch.setattr(
        snapshot_prompts,
        "exact_chat_prompt",
        lambda source, ctx, suffix: seen.append((source, ctx, suffix)) or source,
    )
    assert tb.concurrent_prompt(0, 0, 1024) != tb.concurrent_prompt(1, 0, 1024)
    assert all(ctx == 1024 for _, ctx, _ in seen)


def test_concurrent_records_all_eight_requests(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tb, "concurrent_prompt", lambda i, trial, ctx, variant=0, run=0: str(i)
    )
    monkeypatch.setattr(
        tb,
        "send",
        lambda *a: dict(
            ct=256,
            finish="length",
            dec_tps=20,
            ttft_s=1,
            cached=0,
            pt=1024,
            sha="a",
            _text="x",
            total_s=2.0,
            prompt_ids_sha="p",
            temperature=0.0,
            stop=None,
        ),
    )
    monkeypatch.setattr(tb, "prompt_file_sha256", lambda *_a, **_k: "0" * 64)
    out = tmp_path / "rows.jsonl"
    with out.open("w") as f:
        tb.part_conc(
            SimpleNamespace(url="fixture", model="fixture"),
            f,
            SimpleNamespace(
                conc_ns="8", conc_trials=1, smoke=False, conc_ctx=1024, conc_tokens=256
            ),
        )
    row = json.loads(out.read_text())
    assert len(row["per_req_dec"]) == len(row["ttfts"]) == 8
    assert row["total_tokens"] == 2048
    assert row["agg_tps"] > 0


def test_llama_concurrency_has_per_slot_context(tmp_path):
    launch = tb.bench_engines.build_launch("llamacpp-mtp", 18996, tmp_path, {}, 5120, 8)
    assert launch.cmd[launch.cmd.index("-c") + 1] == "40960"
    assert launch.cmd[launch.cmd.index("-np") + 1] == "8"


def test_extra_recipe_expands_private_home_and_slots(tmp_path):
    be = load("bench_engines")
    extra = load("bench016_extra")
    manifest = {
        "extra_engines": {
            "fixture": dict(
                product="fixture",
                mode="dflash",
                checkpoint="/model",
                argv=["/binary", "--slots", "{parallel}"],
                env={"KAPITSA_HOME": "{home}"},
                engaged_regex="draft",
                startup_ready_regex="program ready",
            )
        }
    }
    extra.install(be, manifest)
    launch = be.build_launch("fixture", 18996, tmp_path, {}, 5120, 8)
    assert launch.cmd[-1] == "8"
    assert launch.env["KAPITSA_HOME"] == str(tmp_path)
    assert be.ENGINES["fixture"].extra["startup_ready_regex"] == "program ready"


def test_summary_retains_three_reps_and_concurrency_metrics(tmp_path):
    for rep in range(3):
        rows = [
            dict(
                part="conc",
                engine="fixture",
                snapshot_rep=rep,
                agg_tps=80 + rep,
                per_req_dec=[10 + rep] * 8,
                ttfts=[1 + rep / 10] * 8,
            ),
            dict(part="part_done", complete=True),
        ]
        (tmp_path / f"rep{rep}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    table = gt.summary(tmp_path)
    assert "aggregate tok/s | 81.000" in table
    assert "per-request decode tok/s" in table
    assert "per-request TTFT s" in table
    assert "3+ reps" in table


def test_splash_gpu_arm_is_explicit(tmp_path):
    be = load("bench_engines")
    gt.configure(
        be,
        dict(
            binaries={},
            engines={"splash-latest": dict(product="Splash fixture", version="1.4.0")},
        ),
    )
    launch = be.build_launch("splash-noane", 18996, tmp_path, {}, 36864, 1)
    assert "--disable-ane" in launch.cmd
    assert (
        "--disable-ane"
        not in be.build_launch("splash-latest", 18996, tmp_path, {}, 36864, 1).cmd
    )


def test_failed_cell_is_not_summarized_even_if_tfbench_completed(tmp_path):
    row = dict(
        part="decode",
        phase="warm",
        ctx=32768,
        kind="code",
        cached=0,
        engine="fixture",
        snapshot_rep=0,
        ttft_s=30,
    )
    (tmp_path / "failed.jsonl").write_text(
        "\n".join(
            map(
                json.dumps,
                [
                    row,
                    dict(part="part_done", complete=True),
                    dict(part="failure", error="invalid warm"),
                ],
            )
        )
    )
    assert "fixture" not in gt.summary(tmp_path)


def test_competitor_research_knobs_are_not_inherited():
    assert tb.bench_engines.scrubbed_env(
        {
            "TF_Q27_COPY_ROWS": "48",
            "NI_KFLASH_BUDGET": "1",
            "KAPITSA_HOME": "/live",
            "PATH": "/bin",
        },
        {"KAPITSA_HOME": "/private"},
    ) == {"PATH": "/bin", "KAPITSA_HOME": "/private"}


def test_cell_has_no_idle_hold_and_cleans_its_private_tmp(monkeypatch, tmp_path):
    import os

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    out = tmp_path / "results"
    seen = []

    def main():
        args = tb.parse_args(sys.argv[1:])
        seen.append(args.idle_s)
        scratch = Path(os.environ["TMPDIR"])
        assert scratch == out / "scratch/tmp"
        (scratch / "cache").write_text("temporary")
        Path(args.out).write_text(
            '{"part":"decode","phase":"cold","cached":0}\n{"part":"part_done","complete":true}\n'
        )

    monkeypatch.setattr(gt, "configure", lambda *a: None)
    monkeypatch.setitem(sys.modules, "tfbench", SimpleNamespace(D="draft", main=main))
    monkeypatch.setattr(sys, "argv", ["fixture"])
    gt.run(
        SimpleNamespace(
            src=str(ROOT),
            out=str(out),
            manifest=str(manifest),
            smoke=False,
            part="decode",
            ctx=1024,
            kind="code",
            engine="tf-native",
            rep=0,
            dry_run=False,
        )
    )
    assert seen == [0]
    assert not (out / "scratch").exists()
    assert len(list(out.glob("*.jsonl"))) == 1


def test_accelerator_startup_waits_for_ready_log(monkeypatch, tmp_path):
    log = tmp_path / "server.log"
    log.write_text("NI_ANE: compiling in the background")
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        log.write_text("NI_ANE: Neural Engine program ready")

    monkeypatch.setattr(tb.time, "sleep", sleep)
    tb.wait_startup_ready(
        SimpleNamespace(poll=lambda: None), log, "Neural Engine program ready"
    )
    assert sleeps == [0.2]
    with pytest.raises(RuntimeError, match="exited"):
        tb.wait_startup_ready(SimpleNamespace(poll=lambda: 1), log, "ready")


def test_accelerator_startup_times_out(monkeypatch, tmp_path):
    now = iter((100, 102))
    monkeypatch.setattr(tb.time, "monotonic", lambda: next(now))
    with pytest.raises(RuntimeError, match="not ready"):
        tb.wait_startup_ready(
            SimpleNamespace(poll=lambda: None), tmp_path / "unused", "ready", timeout=1
        )


def _write_result(
    out, engine, *, sha="a" * 64, pt=1024, phase="cold", ctx=1024, kind="code", rep=0
):
    """Synthetic completed result file: one decode row (prompt fields optional) plus part_done."""
    row = dict(
        engine=engine,
        part="decode",
        ctx=ctx,
        kind=kind,
        phase=phase,
        snapshot_rep=rep,
        ct=512,
        finish="length",
        ttft_s=0.5,
        dec_tps=60.0,
    )
    if pt is not None:
        row["prompt_tokens"] = pt
    if sha is not None:
        row["prompt_sha256"] = sha
    rows = [row, dict(part="part_done", complete=True, engine=engine)]
    path = Path(out) / f"{engine}-decode-{ctx}-{kind}-r{rep}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_prompt_check_passes_when_engines_sent_the_same_prompt(tmp_path):
    _write_result(tmp_path, "yunshu-main", sha="s1", pt=1024)
    _write_result(tmp_path, "tf-native", sha="s1", pt=1025)
    assert gt.prompt_problems(tmp_path) == []


def test_prompt_check_fails_and_names_cells_on_different_prompt(tmp_path):
    _write_result(tmp_path, "yunshu-main", sha="s1", pt=1024)
    _write_result(tmp_path, "omlx-latest", sha="s2", pt=1024)
    problems = gt.prompt_problems(tmp_path)
    assert len(problems) == 1
    assert "mismatch" in problems[0] and "1024 code cold" in problems[0]
    assert "omlx-latest-decode-1024-code-r0.jsonl" in problems[0]


def test_prompt_check_fails_on_prompt_tokens_over_one_percent(tmp_path):
    _write_result(tmp_path, "yunshu-main", sha="s1", pt=1000)
    _write_result(tmp_path, "tf-native", sha="s1", pt=1020)
    assert any("prompt_tokens differ" in p for p in gt.prompt_problems(tmp_path))


def test_prompt_check_fails_closed_when_fields_missing(tmp_path):
    _write_result(tmp_path, "yunshu-main", sha="s1", pt=1024)
    _write_result(tmp_path, "tf-native", sha="s1", pt=None)
    _write_result(tmp_path, "omlx-latest", sha=None, pt=1024)
    problems = gt.prompt_problems(tmp_path)
    assert any(
        "tf-native-decode-1024-code-r0.jsonl" in p and "prompt_tokens missing" in p
        for p in problems
    )
    assert any(
        "omlx-latest-decode-1024-code-r0.jsonl" in p and "prompt_sha256 missing" in p
        for p in problems
    )


def test_prompt_check_ignores_turn2_and_single_engine_groups(tmp_path):
    # turn2 is built from each engine's own cold reply, so differing turn2 prompts are expected.
    _write_result(tmp_path, "yunshu-main", sha="s1", pt=1024, phase="turn2")
    _write_result(tmp_path, "tf-native", sha="s2", pt=1024, phase="turn2")
    _write_result(tmp_path, "solo", sha="x1", pt=4096, ctx=4096)
    assert gt.prompt_problems(tmp_path) == []


def test_summary_action_exits_nonzero_on_prompt_mismatch(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "m.json"
    manifest.write_text("{}")
    _write_result(tmp_path, "yunshu-main", sha="s1", pt=1024)
    _write_result(tmp_path, "omlx-latest", sha="s2", pt=1024)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gaptable.py",
            "summary",
            "--manifest",
            str(manifest),
            "--out",
            str(tmp_path),
            "--src",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        gt.main()
    assert exc.value.code == 1
    assert "PROMPT CHECK FAILED" in capsys.readouterr().err


def test_prompt_digest_is_canonical_and_text_sensitive():
    msgs = [{"role": "user", "content": "hello"}]
    assert tb.prompt_digest(msgs) == tb.prompt_digest(
        [{"content": "hello", "role": "user"}]
    )
    assert tb.prompt_digest(msgs) != tb.prompt_digest(
        [{"role": "user", "content": "hello!"}]
    )
