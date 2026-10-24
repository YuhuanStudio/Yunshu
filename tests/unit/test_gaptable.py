"""CPU fixtures for the public cross-engine harness (no MLX)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    monkeypatch.setattr(tb, "concurrent_prompt", lambda i, trial, ctx: str(i))
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
        ),
    )
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
            )
        }
    }
    extra.install(be, manifest)
    launch = be.build_launch("fixture", 18996, tmp_path, {}, 5120, 8)
    assert launch.cmd[-1] == "8"
    assert launch.env["KAPITSA_HOME"] == str(tmp_path)


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
