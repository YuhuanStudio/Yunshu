"""The direct-capacity research arm preserves prefix bytes and cache ownership."""

import importlib.util
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_vlm.apc_adapters import clone_cache_entry  # noqa: E402
from mlx_vlm.models.cache import KVCache  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "restore_probe",
    Path(__file__).resolve().parents[2] / "scripts/research/probe_prefill_restore.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def cpu_arrays():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


@pytest.mark.parametrize("prefix,capacity", [(32, 33), (256, 257), (257, 1025)])
def test_direct_capacity_is_detached_and_matches_adapter(prefix, capacity):
    source = KVCache()
    source.update_and_fetch(
        mx.arange(prefix * 4).reshape(1, 1, prefix, 4).astype(mx.bfloat16),
        mx.ones((1, 1, prefix, 2), mx.bfloat16) * 3,
    )
    targets = []

    def no_exact_clone(*args, **kwargs):
        raise AssertionError("direct capacity restore must bypass exact cloning")

    actual = probe.direct_clone(
        source,
        min_capacity_tokens=capacity,
        eval_targets=targets,
        clone=clone_cache_entry,
    )
    expected = clone_cache_entry(
        source, min_capacity_tokens=capacity, eval_targets=targets
    )
    source.keys[..., :prefix, :] = -1
    mx.eval(targets)
    assert actual.offset == expected.offset == prefix
    assert actual.meta_state == expected.meta_state
    assert actual.keys.shape == expected.keys.shape
    assert mx.array_equal(actual.keys, expected.keys).item()
    assert mx.array_equal(actual.values, expected.values).item()
    actual.values[..., :prefix, :] = 9
    assert mx.all(source.values[..., :prefix, :] == 3).item()


def test_custom_contract_stays_on_adapter():
    source = object()
    calls = []

    def adapter(c, **kwargs):
        calls.append((c, kwargs))
        return "fallback"

    targets = []
    assert (
        probe.direct_clone(
            source, min_capacity_tokens=1024, eval_targets=targets, clone=adapter
        )
        == "fallback"
    )
    assert calls == [(source, dict(min_capacity_tokens=1024, eval_targets=targets))]


def test_async_restore_keeps_store_synchronous_and_native_restore_isolated():
    from mlx_vlm import apc

    module_spec = importlib.util.spec_from_file_location(
        "async_restore",
        Path(__file__).resolve().parents[2] / "scripts/research/async_restore.py",
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    original = apc._clone_prompt_cache_for_apc
    counts, uninstall = module.install()
    try:
        source = KVCache()
        source.update_and_fetch(mx.ones((1, 1, 8, 4)), mx.ones((1, 1, 8, 4)) * 2)
        mx.eval(source.state)
        stored = apc._clone_prompt_cache_for_apc([source])
        assert counts["async_clones"] == 0
        restored = apc._clone_prompt_cache_for_apc(stored, min_capacity_tokens=80)
        assert counts["async_clones"] == 1
        restored[0].keys[..., :8, :] = 9
        assert mx.all(stored[0].keys[..., :8, :] == 1).item()
        assert restored[0].offset == 8
        assert restored[0].keys.shape[2] == 256
        counts["enabled"] = False
        apc._clone_prompt_cache_for_apc(stored, min_capacity_tokens=80)
        assert counts["async_clones"] == 1
    finally:
        uninstall()
    assert apc._clone_prompt_cache_for_apc is original


def select_m5_sources(monkeypatch):
    from yunshu_engine.kernels.tensorfold import (
        lane_m5,
        lane_qmm,
        lane_widen,
        lane_widen_m5,
    )

    monkeypatch.setattr(lane_qmm, "_resolve_variant", lambda: "m5")
    monkeypatch.setattr(lane_qmm, "_MAIN", lane_m5._MAIN)
    monkeypatch.setattr(lane_qmm, "_MAIN_TILED", lane_m5._MAIN)
    monkeypatch.setattr(lane_widen, "NIBBLES", lane_widen_m5.NIBBLES)
    monkeypatch.setattr(lane_widen, "BYTES", lane_widen_m5.BYTES)


def test_final_barrier_transform_preserves_publication_and_restores_sources(
    monkeypatch,
):
    from yunshu_engine.kernels.tensorfold import lane_qmm, lane_widen

    select_m5_sources(monkeypatch)
    module_spec = importlib.util.spec_from_file_location(
        "lane_final_barrier",
        Path(__file__).resolve().parents[2] / "scripts/research/lane_final_barrier.py",
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    targets = [
        (lane_qmm, "_MAIN"),
        (lane_qmm, "_MAIN_TILED"),
        (lane_widen, "NIBBLES"),
        (lane_widen, "BYTES"),
    ]
    before = [getattr(obj, name) for obj, name in targets]
    uninstall = module.install()
    try:
        for (obj, name), source in zip(targets, before, strict=True):
            transformed = getattr(obj, name)
            assert "if (t + 1 < TMR) threadgroup_barrier" in transformed
            assert transformed.count("threadgroup_barrier") == source.count(
                "threadgroup_barrier"
            )
            assert transformed.replace("if (t + 1 < TMR) ", "") == source
    finally:
        uninstall()
    assert [getattr(obj, name) for obj, name in targets] == before


def test_paired32_keeps_decode_non4bit_and_coop_on_shipped_schedule(monkeypatch):
    from types import SimpleNamespace

    from yunshu_engine.kernels.tensorfold import lane_qmm

    select_m5_sources(monkeypatch)
    module_spec = importlib.util.spec_from_file_location(
        "lane_paired32",
        Path(__file__).resolve().parents[2] / "scripts/research/lane_paired32.py",
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    calls = []

    def matmul(x, weight, sbt, **kwargs):
        calls.append(kwargs)
        return "result"

    monkeypatch.setattr(lane_qmm, "lane_matmul", matmul)
    monkeypatch.setattr(lane_qmm, "weight_bits", lambda weight, k: weight)
    uninstall = module.install()
    try:
        for rows, bits, nt in ((66, 4, 32), (1, 4, 32), (66, 8, 32), (66, 4, 64)):
            x = SimpleNamespace(size=rows * 1024, shape=(rows, 1024))
            assert lane_qmm.lane_matmul(x, bits, None, nt=nt) == "result"
        assert calls == [
            {"nt": 32, "row_block": 64},
            {"nt": 32},
            {"nt": 32},
            {"nt": 64},
        ]
        assert "16 * ((TMR < 2) ? TMR : 2)" in lane_qmm._MAIN_TILED
        assert "C[tb + t][i]" in lane_qmm._MAIN_TILED
    finally:
        uninstall()
    assert lane_qmm.lane_matmul is matmul


@pytest.mark.parametrize("capacity", [8, 80])
def test_cow_restore_matches_adapter_and_isolates_both_directions(capacity):
    from mlx_vlm import apc_adapters
    from mlx_vlm.models.cache import ArraysCache

    module_spec = importlib.util.spec_from_file_location(
        "cow_restore",
        Path(__file__).resolve().parents[2] / "scripts/research/cow_restore.py",
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 1, 8, 4)), mx.ones((1, 1, 8, 4)) * 2)
    rec = ArraysCache(2)
    rec.cache = [mx.ones((1, 2, 3)), None]
    rec.lengths, rec.left_padding = mx.array([8]), mx.array([0])
    mx.eval(kv.state, rec.state, rec.lengths, rec.left_padding)
    original = apc_adapters.clone_cache_entry
    targets = []
    expected = [
        original(c, min_capacity_tokens=capacity, eval_targets=targets)
        for c in (kv, rec)
    ]
    counts, uninstall = module.install()
    try:
        actual = [
            apc_adapters.clone_cache_entry(
                c, min_capacity_tokens=capacity, eval_targets=targets
            )
            for c in (kv, rec)
        ]
        mx.eval(targets)
        assert counts["view_restores"] == 2
        assert actual[0].meta_state == expected[0].meta_state
        assert mx.array_equal(actual[0].keys, expected[0].keys).item()
        assert mx.array_equal(actual[1].cache[0], expected[1].cache[0]).item()
        assert actual[0].keys is not kv.keys
        assert actual[1].cache[0] is not rec.cache[0]
        actual[0].values[..., :8, :] = 9
        actual[1].cache[0][...] = 9
        actual[1].lengths[...] = 20
        assert mx.all(kv.values[..., :8, :] == 2).item()
        assert mx.all(rec.cache[0] == 1).item()
        assert rec.lengths.item() == 8
        kv.keys[..., :8, :] = -1
        rec.left_padding[...] = 4
        assert mx.all(actual[0].keys[..., :8, :] == 1).item()
        assert actual[1].left_padding.item() == 0
        apc_adapters.clone_cache_entry(rec, min_capacity_tokens=None, eval_targets=[])
        assert counts["view_restores"] == 2
    finally:
        uninstall()
    assert apc_adapters.clone_cache_entry is original


@pytest.mark.parametrize(
    "mode", ["shipped", "async", "cow", "cowasync", "spans", "barrier", "paired32"]
)
def test_modern_restore_modes_parse_without_loading_model(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--out",
            str(tmp_path / "unused.jsonl"),
            "--modes",
            "shipped",
            mode,
            "--dry-run",
        ],
    )
    probe.main()


def test_modern_and_legacy_modes_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--out",
            str(tmp_path / "unused.jsonl"),
            "--modes",
            "shipped",
            "direct",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        probe.main()


def test_restore_probe_rejects_ssd_even_for_tiny_model(monkeypatch, tmp_path):
    from yunshu_engine import settings

    monkeypatch.setattr(settings, "get_bool", lambda name: True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--out",
            str(tmp_path / "unused.jsonl"),
            "--model",
            "tiny",
            "--modes",
            "shipped",
        ],
    )
    with pytest.raises(RuntimeError, match="RAM-only APC for every model"):
        probe.main()
