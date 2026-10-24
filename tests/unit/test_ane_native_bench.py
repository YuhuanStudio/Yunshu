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
