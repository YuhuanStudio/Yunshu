"""CPU-only interference harness contracts; no model or MLX import."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def bench():
    path = Path(__file__).resolve().parents[2] / "scripts/research/anenative_tfbench.py"
    spec = importlib.util.spec_from_file_location("anenative_tfbench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapter_never_persists_content_and_keeps_timing(bench, capsys):
    def send(*args, **kwargs):
        assert kwargs["capture_timestamps"] is True
        return {
            "_text": "private reply",
            "head": "private head",
            "_first_at_s": 1.0,
            "sha": "digest",
        }

    fake = SimpleNamespace(
        send=send, META={"engine": "yunshu"}, was_contended=lambda: False
    )
    bench.install_adapter(fake)
    result = fake.send("url", {})
    assert result["_text"] == "private reply" and result["_end_at_s"] > 1
    stream = io.StringIO()
    fake.emit(
        stream,
        **result,
        text="private reply",
        xy={"content": "private nested", "decode_ms": 2.0},
    )
    record = json.loads(stream.getvalue())
    assert record["sha"] == "digest" and record["xy"] == {"decode_ms": 2.0}
    assert "private" not in stream.getvalue() + capsys.readouterr().out


def test_receipt_rejects_cached_or_unfinished_or_unengaged(bench, tmp_path):
    path = tmp_path / "receipt.jsonl"
    rows = [{"part": "session", "engaged_spec_mode": "dflash"}]
    rows += [
        {
            "part": "decode",
            "kind": kind,
            "cached": 0,
            "phase": "cold",
            "ct": 512,
            "finish": "length",
            "sha": "same",
            "_first_at_s": 1.0,
            "_end_at_s": 3.0,
        }
        for kind in ["prose", "code"]
    ]
    rows.append({"complete": True})

    def save():
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    save()
    assert len(bench.read_receipt(path)) == 2
    rows[1]["cached"] = 1
    save()
    with pytest.raises(ValueError, match="cached"):
        bench.read_receipt(path)
    rows[1]["cached"] = 0
    rows[0]["engaged_spec_mode"] = "mtp"
    save()
    with pytest.raises(ValueError, match="engagement"):
        bench.read_receipt(path)
    rows[0]["engaged_spec_mode"] = "dflash"
    rows.pop()
    save()
    with pytest.raises(ValueError, match="complete"):
        bench.read_receipt(path)


def test_rate_only_counts_decode_window_overlap(bench):
    samples = [
        {"start_s": 0.0, "end_s": 1.0, "calls": 10},
        {"start_s": 1.0, "end_s": 2.0, "calls": 20},
    ]
    assert bench.rate_in_window(samples, 0.5, 1.5) == 15.0
    assert bench.rate_in_window(samples, 3.0, 4.0) is None
    with pytest.raises(ValueError):
        bench.rate_in_window([{"start_s": 2.0, "end_s": 1.0, "calls": 1}], 0.0, 3.0)


def test_overlap_parser_and_fail_fast_slice_validation():
    path = Path(__file__).resolve().parents[2] / "scripts/research/anenative_overlap.py"
    spec = importlib.util.spec_from_file_location("anenative_overlap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parser().parse_args(
        ["--cache", "/Volumes/P5Plus/cache", "--out", "/Volumes/P5Plus/out.jsonl"]
    )
    assert args.rows == 1024 and args.mode == "fp16"
    assert module.validate_channels("1088,2176", 17408) == [1088, 2176]
    for bad in ["0", "63", "17408", "-64"]:
        with pytest.raises(ValueError):
            module.validate_channels(bad, 17408)


def test_alias_probe_import_and_parser_are_cpu_only():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/anenative_alias_probe.py"
    )
    spec = importlib.util.spec_from_file_location("anenative_alias_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parser().parse_args(["--cache", "/Volumes/P5Plus/cache"])
    assert (args.height, args.width) == (64, 128)


def test_shared_probe_parser_is_cpu_only():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/anenative_shared_probe.py"
    )
    spec = importlib.util.spec_from_file_location("anenative_shared_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parser().parse_args(["--cache", "/Volumes/P5Plus/cache"])
    assert args.loops == 8


def test_coreml_plan_counts_nested_device_assignments():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/anenative_coreml_plan.py"
    )
    spec = importlib.util.spec_from_file_location("anenative_coreml_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    child = SimpleNamespace(operator_name="linear", blocks=[])
    parent = SimpleNamespace(
        operator_name="region", blocks=[SimpleNamespace(operations=[child])]
    )

    class Device:
        pass

    plan = SimpleNamespace(
        model_structure=SimpleNamespace(
            program=SimpleNamespace(
                functions={
                    "main": SimpleNamespace(block=SimpleNamespace(operations=[parent]))
                }
            )
        ),
        get_compute_device_usage_for_mlprogram_operation=lambda op: (
            None if op is parent else SimpleNamespace(preferred_compute_device=Device())
        ),
    )
    result = module.placement(plan)
    assert result["preferred_counts"] == {"unassigned_or_constant": 1, "Device": 1}


def test_handoff_parser_rejects_empty_or_unaligned_gpu_slice():
    path = Path(__file__).resolve().parents[2] / "scripts/research/anenative_handoff.py"
    spec = importlib.util.spec_from_file_location("anenative_handoff", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.valid_shape(64, 2176, 17408) == 128
    assert module.valid_shape(1024, 2176, 17408) == 1024
    for rows, channels in [(0, 2176), (1024, 0), (1024, 63), (1024, 17408)]:
        with pytest.raises(ValueError):
            module.valid_shape(rows, channels, 17408)
    args = module.parser().parse_args(
        ["--cache", "/Volumes/P5Plus/cache", "--out", "/Volumes/P5Plus/out.jsonl"]
    )
    assert args.rows == 1024 and args.channels == 2176


def test_fault_probe_parser_does_not_load_private_runtime():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/anenative_fault_probe.py"
    )
    spec = importlib.util.spec_from_file_location("anenative_fault_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parser().parse_args(
        [
            "--library",
            "/Volumes/P5Plus/test.dylib",
            "--cache",
            "/Volumes/P5Plus/cache",
            "--fault",
            "stalled",
        ]
    )
    assert args.fault == "stalled"


def test_grid_oracle_compares_bits_including_signed_zero():
    import numpy as np

    path = (
        Path(__file__).resolve().parents[2] / "scripts/research/anenative_grid_probe.py"
    )
    spec = importlib.util.spec_from_file_location("anenative_grid_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    left = np.array([0.0, 1.0], dtype=np.float16)
    right = np.array([-0.0, 1.0], dtype=np.float16)
    assert module.bit_difference(left, right) == 1
    with pytest.raises(ValueError):
        module.bit_difference(left, np.array([np.nan, 1.0], dtype=np.float16))


def test_warm_interference_requires_equal_hot_cache_and_digest():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/anenative_warm_interference.py"
    )
    spec = importlib.util.spec_from_file_location("anenative_warm_interference", path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    row = {
        "ct": 512,
        "finish": "length",
        "cached": 32768,
        "sha": "abc",
        "xy": {"speculative": {"mode": "dflash"}},
    }
    cache = {}
    module.validate_row(row, "code", cache, {"code": "abc"})
    assert cache == {"code": 32768}
    for change in [{"cached": 0}, {"cached": 32704}, {"sha": "other"}, {"ct": 128}]:
        with pytest.raises(ValueError):
            module.validate_row({**row, **change}, "code", cache, {"code": "abc"})


def test_staging_relief_only_releases_unreachable_allocator_memory(monkeypatch):
    from unittest.mock import Mock

    path = Path(__file__).resolve().parents[2] / "scripts/research/anenative_load.py"
    spec = importlib.util.spec_from_file_location("anenative_load_trim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trim = Mock(return_value=4096)
    collect = Mock()
    monkeypatch.setattr(module.gc, "collect", collect)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(
        module.c, "CDLL", lambda _: SimpleNamespace(malloc_zone_pressure_relief=trim)
    )
    assert module.release_staging() == 4096
    collect.assert_called_once()
    trim.assert_called_once_with(None, 0)


def test_warm_summary_requires_live_load_and_complete_equal_arms(tmp_path):
    import sys

    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/anenative_warm_summary.py"
    )
    spec = importlib.util.spec_from_file_location("anenative_warm_summary", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    (tmp_path / "complete.json").write_text('{"complete":true}')
    worker = {
        "complete": True,
        "rows": 128,
        "whole_mlp": True,
        "rate_samples": [{"start_s": 1.0, "end_s": 3.0, "calls": 20, "active": True}],
    }
    for backend in ("native", "coreml"):
        (tmp_path / f"{backend}.json").write_text(json.dumps(worker))
    rows = []
    backends = [
        "off",
        "native",
        "native",
        "off",
        "coreml",
        "coreml",
        "off",
        "native",
        "coreml",
        "off",
    ]
    for kind in ("prose", "code"):
        for arm, backend in enumerate(backends):
            rows.append(
                {
                    "stage": "decode",
                    "kind": kind,
                    "arm": str(arm),
                    "backend": backend,
                    "ct": 512,
                    "finish": "length",
                    "sha": "same",
                    "cached": 32768,
                    "xy": {"speculative": {"mode": "dflash"}},
                    "dec_tps": 100 if backend == "off" else 90,
                    "_first_at_s": 1.0,
                    "_end_at_s": 3.0,
                }
            )
    rows.append({"complete": True})
    receipt = tmp_path / "decode.jsonl"
    receipt.write_text("\n".join(json.dumps(r) for r in rows))
    result = module.summarize(tmp_path)
    assert len(result["cells"]) == 4
    assert result["cells"][0]["decode_slowdown_percent"] == pytest.approx(10)
    assert result["cells"][0]["mlp_only_model64_tps_upper_bound"] == 20
    worker["rate_samples"][0]["active"] = False
    (tmp_path / "native.json").write_text(json.dumps(worker))
    with pytest.raises(ValueError, match="missing"):
        module.summarize(tmp_path)
