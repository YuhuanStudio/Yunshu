"""CPU-only fixtures for the served TTFT probe; never import MLX."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts/research/ttft_warm_probe.py"
_SPEC = importlib.util.spec_from_file_location("ttft_warm_probe", _PATH)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_stage_uses_inner_frame_before_outer_prefill():
    assert probe.stage(["apc.py:clone:12", "prefill.py:run:8"]) == "apc_lookup_clone"
    assert probe.stage(["sampler.py:sample:3", "mtp_lane.py:step:8"]) == "sampling"


def test_stage_summary_integrates_one_thread_and_preserves_missing_time():
    samples = [
        (1.0, "mlx_0", ["apc.py:clone:1"]),
        (1.002, "mlx_0", ["prefill.py:run:1"]),
        (1.001, "http", ["apc.py:clone:1"]),
    ]
    result = probe.stage_summary(samples, 0.999, 1.005, "mlx")
    assert result["apc_lookup_clone"] == 2
    assert result["new_token_prefill"] == 2
    assert result["unobserved"] == 2
    assert sum(result.values()) == 6


def test_sample_gap_is_not_charged_to_gpu_work():
    result = probe.stage_summary([(0, "mlx", ["prefill.py:run:1"])], 0, 1, "mlx")
    assert result["new_token_prefill"] == 2
    assert result["unobserved"] == 998


@pytest.mark.parametrize("phase,cached", [("cold", 2), ("warm", 0), ("followup", 0)])
def test_first_request_validation_fails_fast(phase, cached):
    with pytest.raises(ValueError):
        probe.validate_response(dict(ct=1, ttft_s=0.1, cached=cached, pt=100), phase)


def test_valid_phases():
    for phase, cached in [("cold", 0), ("warm", 100), ("followup", 90)]:
        probe.validate_response(dict(ct=1, ttft_s=0.1, cached=cached, pt=100), phase)


def test_call_spans_records_gpu_wait_without_importing_mlx(monkeypatch):
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "ttft_spans", _PATH.parent / "ttft_probe_site/spans.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    times = iter((1.0, 1.1))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(times))
    frame = SimpleNamespace(
        f_code=SimpleNamespace(co_filename="ar.py", co_name="generate"), f_lineno=2302
    )
    call = SimpleNamespace(__name__="eval", __module__="mlx.core")
    collector = module.CallSpans()
    collector.profile(frame, "c_call", call)
    collector.profile(frame, "c_return", call)
    row = collector.rows[0]
    assert row[:2] == (1.0, 1.1)
    assert row[3] == "gpu_wait"
    assert row[4] == "ar.py:generate:2302:eval"
    assert not collector.active


def test_timeline_collapses_runs_and_drops_short_ones():
    s = [(1.000 + 0.002 * i, "mlx", ["a", "b"]) for i in range(10)]
    s += [(1.020 + 0.002 * i, "mlx", ["c"]) for i in range(2)]
    s += [(1.5, "mlx", ["late"])]
    out = probe.timeline(s, 1.0, 1.1)
    assert list(out) == ["mlx"]
    assert [r[2] for r in out["mlx"]] == ["a < b"]
    assert out["mlx"][0][0] == 0.0


def test_load_samples_reads_every_pid_file(tmp_path):
    for pid in (1, 2):
        (tmp_path / f"samples.{pid}").write_text(
            json.dumps([float(pid), "t", ["x"]]) + "\n"
        )
    assert len(probe.load_samples(str(tmp_path / "samples"))) == 2


def test_sampler_roundtrip_in_a_real_child_process(tmp_path):
    import os
    import subprocess
    import sys
    import time

    site = Path(probe.HERE) / "ttft_probe_site"
    env = dict(
        os.environ, PYTHONPATH=str(site), TTFT_PROBE_OUT=str(tmp_path / "samples")
    )
    child = subprocess.Popen(
        [sys.executable, "-c", "import time\nwhile True: time.sleep(0.01)"],
        env=env,
        start_new_session=True,
    )
    try:
        time.sleep(2)
        probe.signal_dump(child.pid)
        time.sleep(1)
        rows = probe.load_samples(str(tmp_path / "samples"))
        assert rows and rows[0][1] == "MainThread"
        t = rows[-1][0]
        assert probe.timeline(rows, t - 0.3, t)
    finally:
        child.kill()


def test_nested_call_spans_are_not_double_counted():
    spans = [
        dict(start_ms=10, end_ms=90, thread="mlx", stage="first_round"),
        dict(start_ms=20, end_ms=30, thread="mlx", stage="sampling"),
        dict(start_ms=40, end_ms=80, thread="mlx", stage="gpu_wait"),
    ]
    result = probe.partition_spans(spans, 100, "mlx")
    assert result == dict(first_round=70, sampling=10, http_sse_queue_other=20)
    assert sum(result.values()) == 100


def test_main_with_fake_server_writes_complete_and_engaged_mode(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    log = tmp_path / "server.log"
    log.write_text("fake server")
    server = SimpleNamespace(
        model="fake",
        url="fake",
        proc=SimpleNamespace(pid=1),
        log=log,
        kill=lambda: None,
    )
    requests = []

    def send(url, body):
        requests.append(body)
        phase = len(requests)
        return dict(
            _text="reply", ct=32, pt=100, cached=0 if phase == 1 else 99, ttft_s=0.1
        )

    fake = SimpleNamespace(
        subprocess=SimpleNamespace(Popen=lambda *a, **k: None),
        Srv=lambda *a, **k: server,
        load_prompt=lambda name: "prompt",
        req=lambda model, prompt, count: dict(
            messages=[dict(role="user", content=prompt)]
        ),
        send=send,
        engaged_spec_mode=lambda *a: "off",
    )
    monkeypatch.setitem(sys.modules, "tfbench", fake)
    monkeypatch.setattr(probe, "signal_dump", lambda pid: None)
    monkeypatch.setattr(probe, "load_samples", lambda prefix: [(0, "mlx", ["fake"])])
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    out = tmp_path / "result.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--model",
            "fake",
            "--src",
            "fake",
            "--tag",
            "cpu",
            "--out",
            str(out),
            "--warm",
            "1",
            "--draft",
            "off",
        ],
    )
    probe.main()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[-1] == dict(complete=True, requests=3, threads=["mlx"])
    assert [row["phase"] for row in rows[:-1]] == ["cold", "warm", "followup"]
    assert all(row["engaged"] == "off" for row in rows[:-1])
    assert requests[-1]["messages"][1] == dict(role="assistant", content="reply")


def test_load_samples_does_not_mix_call_spans_with_stacks(tmp_path):
    (tmp_path / "samples.1").write_text(json.dumps([1, "mlx", ["frame"]]) + "\n")
    (tmp_path / "samples.spans.1").write_text(
        json.dumps([1, 2, "mlx", "gpu_wait", "eval"]) + "\n"
    )
    assert probe.load_samples(str(tmp_path / "samples")) == [[1, "mlx", ["frame"]]]
