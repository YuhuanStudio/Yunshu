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


@pytest.mark.parametrize("gc_policy", ["auto", "defer"])
@pytest.mark.parametrize("mode", ["off", "dflash"])
def test_main_with_fake_server_writes_complete_and_engaged_mode(
    tmp_path, monkeypatch, mode, gc_policy
):
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

    def send(url, body, *, capture_timestamps=False, trace_id=None):
        assert capture_timestamps and trace_id.startswith("ttft-")
        requests.append(body)
        phase = len(requests)
        # All fake requests share the same clock interval. Only header correlation
        # can keep an older request's final coroutine resume out of the next one.
        with (tmp_path / "cpu" / "samples.spans.1").open("a") as f:
            for stamp, label in ((1.01, "http_receive"), (1.08, "http_first_sse")):
                f.write(
                    json.dumps(
                        [
                            stamp,
                            stamp,
                            "MainThread",
                            label,
                            "uvicorn:send:request=" + trace_id,
                        ]
                    )
                    + "\n"
                )
        return dict(
            _text="reply",
            ct=32,
            pt=100,
            cached=0 if phase == 1 else 99,
            ttft_s=0.1,
            _sent_at_s=1.0,
            _first_at_s=1.1,
        )

    def create_server(engine, env, tag, **kw):
        assert env["TTFT_PROBE_COPY_INDEX_GC"] == gc_policy
        (tmp_path / "cpu" / "runtime.1.json").write_text(
            json.dumps({"copy_index_gc": gc_policy})
        )
        assert env["YUNSHU_VLM_DRAFT"] == (
            "/fixture/drafter" if mode == "dflash" else "off"
        )
        return server

    fake = SimpleNamespace(
        subprocess=SimpleNamespace(Popen=lambda *a, **k: None),
        Srv=create_server,
        D="/fixture/drafter",
        load_prompt=lambda name: "prompt",
        req=lambda model, prompt, count: dict(
            messages=[dict(role="user", content=prompt)]
        ),
        send=send,
        engaged_spec_mode=lambda *a: mode,
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
            mode,
            "--copy-index-gc",
            gc_policy,
        ],
    )
    probe.main()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[-1] == dict(complete=True, requests=3, threads=["mlx"])
    assert [row["phase"] for row in rows[:-1]] == ["cold", "warm", "followup"]
    assert all(row["engaged"] == mode for row in rows[:-1])
    assert all(
        row["copy_index_gc"] == gc_policy and row["copy_index_gc_engaged"]
        for row in rows[:-1]
    )
    assert requests[-1]["messages"][1] == dict(role="assistant", content="reply")
    for row in rows[:-1]:
        assert len(row["call_spans"]) == 2
        assert row["http"]["ingress_ms"] == pytest.approx(10)
        assert row["http"]["first_sse_submit_ms"] == pytest.approx(80)
        assert row["http"]["sse_submit_to_client_ms"] == pytest.approx(20)


def test_load_samples_does_not_mix_call_spans_with_stacks(tmp_path):
    (tmp_path / "samples.1").write_text(json.dumps([1, "mlx", ["frame"]]) + "\n")
    (tmp_path / "samples.spans.1").write_text(
        json.dumps([1, 2, "mlx", "gpu_wait", "eval"]) + "\n"
    )
    assert probe.load_samples(str(tmp_path / "samples")) == [[1, "mlx", ["frame"]]]


def test_http_trace_ignores_role_usage_and_done():
    spec = importlib.util.spec_from_file_location(
        "http_spans", _PATH.parent / "ttft_probe_site/spans.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for payload in (
        dict(choices=[dict(delta=dict(role="assistant"))]),
        dict(usage={}),
        "[DONE]",
    ):
        assert not module.sse_has_token(
            b"data: " + json.dumps(payload).encode() + b"\n\n"
        )
    for delta in (
        dict(content="hello"),
        dict(reasoning_content="think"),
        dict(tool_calls=[dict(index=0)]),
    ):
        body = (
            b"data: " + json.dumps(dict(choices=[dict(delta=delta)])).encode() + b"\n\n"
        )
        assert module.sse_has_token(body)
    assert not module.sse_has_token(b"data: [DONE]\n\n")


def test_native_wrapper_preserves_arguments_return_and_records_wait(monkeypatch):
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "native_spans", _PATH.parent / "ttft_probe_site/spans.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    times = iter((1.0, 1.1, 2.0, 2.2))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(times))
    calls = []

    def original(*args, **kw):
        calls.append((args, kw))
        return "result"

    native = SimpleNamespace(eval=original)
    spans = module.CallSpans()
    spans.wrap_native(native)
    inputs = ([1, 2],)
    assert native.eval(*inputs, value=True) == "result"
    assert calls == [(inputs, dict(value=True))]
    assert spans.rows[0][3] == "gpu_wait"
    spans.gc_event("start", dict(generation=2))
    spans.gc_event("stop", dict(generation=2, collected=7))
    assert spans.rows[1][:2] == (2.0, 2.2)
    assert spans.rows[1][3] == "python_gc"
    assert "collected=7" in spans.rows[1][4]


def test_tiny_can_explicitly_allow_followup_miss_but_long_default_fails():
    result = dict(ct=1, ttft_s=0.1, cached=0, pt=100)
    with pytest.raises(ValueError, match="followup"):
        probe.validate_response(result, "followup")
    probe.validate_response(result, "followup", require_followup_hit=False)


def test_clip_spans_excludes_other_requests_and_splits_gc_at_first_token():
    spans = [
        (1.05, 1.15, "mlx", "python_gc", "gen2"),
        (1.19, 1.3, "mlx", "copy_index", "index"),
        (1.21, 1.3, "mlx", "python_gc", "after-response"),
        (1.12, 1.12, "http", "http_first_sse", "request=other"),
    ]
    before = probe.clip_spans(spans, 1.0, 1.1, "this")
    after = probe.clip_spans(spans, 1.1, 1.2, "this")
    assert [r["stage"] for r in before] == ["python_gc"]
    assert before[0]["end_ms"] == pytest.approx(100)
    assert [r["stage"] for r in after] == ["python_gc", "copy_index"]
    assert after[0]["start_ms"] == 0
    assert after[0]["end_ms"] == pytest.approx(50)
    assert after[1]["end_ms"] == pytest.approx(100)


@pytest.mark.parametrize("assignment", ["BROKEN", "TTFT_PROBE_COPY_INDEX_GC=freeze"])
def test_probe_validates_experiment_env_before_server_start(
    tmp_path, monkeypatch, assignment
):
    import sys

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
            "bad",
            "--out",
            str(tmp_path / "out"),
            "--env",
            assignment,
        ],
    )
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2
