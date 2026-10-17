"""CPU gates for the fresh-release snapshot, no MLX import."""

from types import SimpleNamespace

from scripts.research import bench016 as b


def test_admission_rejects_unreadable_swap_and_foreign_residents():
    assert b.admission("vm.swapusage: used = 0.00M", "") == []
    assert b.admission("used = 1.01G", "") == []
    assert b.admission("unknown", "")
    assert b.admission("used = 10M", "123 2097152 /bin/llama-server -m weights")
    assert b.admission("used = 10M", "123 100 /bin/llama-server -m weights") == []


def test_missing_or_short_rows_and_unknown_contention_fail(monkeypatch):
    monkeypatch.setattr(b.bs, "validate_rows", lambda job, rows: [])
    job = SimpleNamespace(stage="cell")
    rows = [
        dict(part="decode", phase=p, ct=n, sha="same")
        for p, n in [("cold", 2048), ("warm", 2048), ("turn2", 256)]
    ]
    assert b.verdict(job, rows, {"rc": 0, "contended": False}) == []
    assert b.verdict(job, rows[:-1], {"rc": 0, "contended": False})
    assert b.verdict(job, rows, {"rc": 0})
    assert b.verdict(job, rows, {"rc": 1, "contended": False})
    rows[1]["sha"] = "different"
    assert b.verdict(job, rows, {"rc": 0, "contended": False})


def test_extra_recipe_has_bounded_port_and_requires_mode_evidence():
    from scripts.research import bench016_extra

    base = b.be.ENGINES["yunshu-main"]
    fake = SimpleNamespace(
        ENGINES={"yunshu-main": base, "fn-yunshu-main": base},
        build_launch=lambda *args: None,
        detect_mode=lambda *args: None,
        required_paths=lambda *args: [],
        Launch=b.be.Launch,
        scrubbed_env=b.be.scrubbed_env,
    )
    cfg = {
        "product": "fixture",
        "checkpoint": "/model",
        "mode": "mtp",
        "argv": ["/external/bin/engine", "--port", "{port}", "--ctx", "{ctx}"],
        "engaged_regex": "MTP head loaded",
    }
    bench016_extra.install(fake, {"extra_engines": {"fixture": cfg}})
    import pytest

    with pytest.raises(ValueError):
        fake.build_launch("fixture", 8000, "/external/home", {})
    launch = fake.build_launch("fixture", 18990, "/external/home", {}, 131072)
    assert launch.cmd[-1] == "131072"
    assert fake.detect_mode("fixture", "MTP requested") is None
    assert fake.detect_mode("fixture", "MTP head loaded") == "mtp"


def test_ollama_environment_is_external_and_owns_only_allowed_port(tmp_path):
    import pytest
    from scripts.research.bench016_ollama import environment

    with pytest.raises(ValueError):
        environment(tmp_path, 11434, 32768)
    env = environment(tmp_path, 18990, 131072)
    assert env["OLLAMA_HOST"] == "127.0.0.1:18990"
    assert env["OLLAMA_MODELS"] == str(tmp_path / "models")
    assert env["OLLAMA_CONTEXT_LENGTH"] == "131072"


def test_pinned_sha_check_applies_only_to_yunshu_arms():
    from scripts.research import bench_engines as be
    from scripts.research import bench_snapshot as snap

    job = SimpleNamespace(
        engine="omlx-latest",
        env={"TFB_EXPECT_YUNSHU_SHA": "abc"},
        stage="cell",
        part="decode",
        ctxs=[512],
        kinds=("prose",),
        name="x",
    )
    if job.engine not in be.ENGINES:
        job.engine = next(k for k, v in be.ENGINES.items() if v.kind != "yunshu")
    rows = [dict(engine=job.engine, git_sha="other")]
    probs = snap.validate_rows(job, rows)
    assert not any("release SHA" in p for p in probs)
