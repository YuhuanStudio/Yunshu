"""CPU-only tests for the fullcmp wiring of tfbench (TFB_FULLCMP=1).

(a) every record carries the fields fullcmp_audit requires; (b) telemetry marks and a required
sampler (failure = error record); (c) the Yunshu pin fails closed; (d) swap state at start and
end. No server, no GPU: the server, the sampler and the HTTP stream are fakes.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_audit as fa  # noqa: E402
import fullcmp_provenance as fp  # noqa: E402
import tfbench  # noqa: E402

PIN_SHA = "0f2960047855546a9f13b2631b680352b1c57cfe"
WEIGHT_SHA = "c" * 64
SHARD = "model-00001-of-00001.safetensors"


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A pinned venv + worktree, a model dir with config and census entry, and a pin file."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("py")
    (venv / "bin" / "yunshu").write_text("entry")
    wt = tmp_path / "wt"
    (wt / "python").mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"quantization": {"bits": 4, "group_size": 64}})
    )
    (model / SHARD).write_text("weights")
    pin = {
        "sha": PIN_SHA,
        "version": "0.1.7.dev0",
        "worktree": str(wt),
        "venv": str(venv),
        "python_executable": str(venv / "bin" / "python"),
    }
    pin_path = tmp_path / "yunshu_pin.json"
    pin_path.write_text(json.dumps(pin))
    census = {
        "yunshu": {
            "latest": "0.1.7.dev0 (repo pin)",
            "latest_date": "2026-11-03 (commit x)",
        },
        "models": {
            "qwen-test": {"path": str(model), "sha256": {SHARD: WEIGHT_SHA}},
        },
    }
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps(census))
    monkeypatch.setattr(fp, "PIN_PATH", pin_path)
    monkeypatch.setattr(fp, "CENSUS_PATH", census_path)
    good = {
        "TFB_YUNSHU_BIN": str(venv / "bin" / "yunshu"),
        "TFB_YUNSHU_SRC": str(wt / "python"),
    }
    return types.SimpleNamespace(
        tmp=tmp_path,
        venv=venv,
        wt=wt,
        model=model,
        pin=pin,
        good=good,
        census=census,
        pin_path=pin_path,
    )


def _sse(*chunks: dict) -> list[bytes]:
    lines = [b"data: " + json.dumps(c).encode() + b"\n" for c in chunks]
    return lines + [b"data: [DONE]\n"]


class _Resp:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return iter(self._lines)

    def __exit__(self, *exc):
        return False


def _stream_ok():
    return _sse(
        {"choices": [{"delta": {"content": "hi"}}]},
        {
            "choices": [{"delta": {}, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    )


@pytest.fixture
def fake_http(monkeypatch):
    calls = []

    def urlopen(req, timeout=None):
        calls.append(req)
        return _Resp(_stream_ok())

    monkeypatch.setattr(tfbench.urllib.request, "urlopen", urlopen)
    return calls


class FakeSampler:
    """Stands in for fullcmp_telemetry.Sampler. FAIL makes start() raise."""

    FAIL = False
    instances: list = []

    def __init__(self, pid, port, out, *a, **k):
        self.pid, self.port, self.out = pid, port, out
        self.marks, self.stopped = [], False
        FakeSampler.instances.append(self)

    def start(self):
        if FakeSampler.FAIL:
            raise RuntimeError("psutil cannot read the server pid")
        Path(self.out).write_text(json.dumps({"part": "telemetry_header"}) + "\n")
        return self

    def mark(self, ev, **fields):
        self.marks.append((ev, fields))

    def stop(self):
        self.stopped = True
        return {}


@pytest.fixture
def fake_telemetry(monkeypatch):
    FakeSampler.FAIL = False
    FakeSampler.instances = []
    mod = types.ModuleType("fullcmp_telemetry")
    mod.Sampler = FakeSampler
    mod.swap_record = lambda: {"total_mb": 1024.0, "used_mb": 77.0, "free_mb": 947.0}
    monkeypatch.setitem(sys.modules, "fullcmp_telemetry", mod)
    monkeypatch.setattr(tfbench, "_ACTIVE_TELEMETRY", None)
    return mod


# ---------------------------------------------------------------- (c) pin
def test_pin_missing_env_refuses(layout):
    with pytest.raises(
        fp.ProvenanceError, match="requires TFB_YUNSHU_BIN and TFB_YUNSHU_SRC"
    ):
        fp.require_yunshu_pin({})
    with pytest.raises(fp.ProvenanceError):
        fp.require_yunshu_pin({"TFB_YUNSHU_BIN": layout.good["TFB_YUNSHU_BIN"]})


def test_pin_mismatched_bin_refuses(layout):
    main_venv_bin = layout.tmp / "main-venv" / "bin" / "yunshu"
    main_venv_bin.parent.mkdir(parents=True)
    main_venv_bin.write_text("main checkout entry")
    env = dict(layout.good, TFB_YUNSHU_BIN=str(main_venv_bin))
    with pytest.raises(fp.ProvenanceError, match="not the pinned venv entry"):
        fp.require_yunshu_pin(env)


def test_pin_mismatched_src_refuses(layout):
    env = dict(layout.good, TFB_YUNSHU_SRC=str(layout.tmp / "other" / "python"))
    with pytest.raises(fp.ProvenanceError, match="not the pinned worktree"):
        fp.require_yunshu_pin(env)


def test_pin_good_env_returns_pin(layout):
    got = fp.require_yunshu_pin(layout.good)
    assert got["version"] == "0.1.7.dev0"
    assert got["git_sha"] == PIN_SHA
    assert got["python"] == layout.pin["python_executable"]


def test_server_never_launched_without_pin(layout, monkeypatch):
    """Srv refuses before any port, directory or process is created (no main-checkout fallback)."""
    monkeypatch.setenv("TFB_FULLCMP", "1")
    monkeypatch.delenv("TFB_YUNSHU_BIN", raising=False)
    monkeypatch.delenv("TFB_YUNSHU_SRC", raising=False)

    def no_popen(*a, **k):
        raise AssertionError("a server process must not start")

    monkeypatch.setattr(tfbench.subprocess, "Popen", no_popen)
    with pytest.raises(fp.ProvenanceError):
        tfbench.Srv("yunshu", {}, "t", model=str(layout.model))


def test_server_refuses_mismatched_path_before_popen(layout, monkeypatch):
    monkeypatch.setenv("TFB_FULLCMP", "1")
    monkeypatch.setenv(
        "TFB_YUNSHU_BIN", str(layout.tmp / "main" / ".venv" / "bin" / "yunshu")
    )
    monkeypatch.setenv("TFB_YUNSHU_SRC", layout.good["TFB_YUNSHU_SRC"])
    monkeypatch.setattr(
        tfbench.subprocess,
        "Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no launch")),
    )
    with pytest.raises(fp.ProvenanceError, match="not the pinned venv entry"):
        tfbench.Srv("yunshu", {}, "t", model=str(layout.model))


def test_pinned_cmd_replaces_main_checkout_entry(layout):
    main_py = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python"
    cmd = [main_py, "-m", "yunshu_cli", "serve", "-m", "M", "--port", "18991"]
    got = fp.pinned_yunshu_cmd(
        cmd,
        layout.pin | {"python": layout.pin["python_executable"]},
        tfbench.bench_engines.yunshu_args,
    )
    assert got[:3] == [layout.pin["python_executable"], "-m", "yunshu_cli"]
    assert main_py not in got


def test_verify_pinned_server_checks_version_and_python(layout):
    pin = fp.require_yunshu_pin(layout.good)
    good_cmd = f"{pin['python']} -m yunshu_cli serve -m M"
    fp.verify_pinned_server({"version": "0.1.7.dev0"}, good_cmd, pin)
    with pytest.raises(fp.ProvenanceError, match="reports version"):
        fp.verify_pinned_server({"version": "0.1.6"}, good_cmd, pin)
    with pytest.raises(fp.ProvenanceError, match="not the pinned python"):
        fp.verify_pinned_server(
            {"version": "0.1.7.dev0"}, "/usr/bin/python3 -m yunshu_cli", pin
        )


# ---------------------------------------------------------------- (a) fields
def test_send_record_carries_request_fields(fake_http):
    body = tfbench.req("M", "hello", 8)
    r = tfbench.send("http://127.0.0.1:1", body)
    for key in (
        "prompt_ids_sha",
        "prompt_sha256",
        "max_tokens",
        "temperature",
        "stop",
        "harness",
        "pt",
        "ct",
        "cached",
        "ttft_s",
        "total_s",
        "dec_tps",
        "finish",
        "sha",
    ):
        assert key in r, key
    assert r["max_tokens"] == 8 and r["temperature"] == 0 and r["stop"] == []
    assert r["harness"] == tfbench.HARNESS_ID
    assert (
        r["prompt_ids_sha"]
        == r["prompt_sha256"]
        == tfbench.prompt_digest(body["messages"])
    )


def test_audit_request_schema_names_match_harness(fake_http):
    """Every request field the audit requires is one the harness emits (decode row)."""
    r = tfbench.send("http://127.0.0.1:1", tfbench.req("M", "hello", 8))
    decode_row = dict(
        r,
        part="decode",
        ctx=1024,
        kind="prose",
        phase="cold",
        cell="c",
        prompt_file="/x/prose-1024.txt",
        contended=False,
    )
    decode_row.pop("_text", None)
    required = set(fa.FULLCMP_SCHEMA["request"]) - {
        "error"
    }  # error: absent on measured rows
    missing = required - set(decode_row)
    assert not missing, sorted(missing)


def test_decode_prompt_file_is_the_real_source():
    assert tfbench.decode_prompt_file("code", 8192).endswith("prompts/code-8192.txt")
    assert tfbench.decode_prompt_file("code", 8192, smoke=True) == "inline:smoke"


def test_cell_name_and_telemetry_path_match_audit(tmp_path):
    out = str(tmp_path / "yunshu" / "prose-1024-r2.jsonl")
    assert tfbench.cell_name(out, 2) == "prose-1024"
    tpath = fp.telemetry_out(out, "yunshu", str(tmp_path / "root"))
    assert tpath == str(tmp_path / "root" / "yunshu" / "prose-1024-r2.telemetry.jsonl")
    # the audit reads the telemetry beside the cell file, named <cell>-rN.telemetry.jsonl
    assert fa.telemetry_path(Path(out)).name == "prose-1024-r2.telemetry.jsonl"


def test_fullcmp_header_has_every_required_header_field(layout, monkeypatch):
    s = types.SimpleNamespace(
        cmd=[layout.pin["python_executable"], "-m", "yunshu_cli", "serve"],
        pin=fp.require_yunshu_pin(layout.good),
        requested_spec_mode="mtp",
        engaged_log_line="VLM batch runner: draft=mtp",
        home=layout.tmp / "home",
    )
    monkeypatch.setattr(
        tfbench.bench_engines,
        "engine_version",
        lambda eng, src="": ("0.1.7.dev0", PIN_SHA),
    )
    a = types.SimpleNamespace(
        engine="yunshu",
        model=str(layout.model),
        rep=0,
        out=str(layout.tmp / "yunshu" / "cell-r0.jsonl"),
    )
    env = dict(layout.good, TFB_RUN_ID="run1", TFB_GROUP="g1", GPUQ_JOB_ID="job1")
    fc = tfbench.fullcmp_header(
        s,
        a,
        env,
        census=layout.census,
        job_start_s=1.0,
        swap_start={"used_mb": 1.0},
        telemetry_path="/t.jsonl",
    )
    # what emit() adds to every session row, plus the start row's explicit fields
    emitted = {
        "part",
        "engine",
        "rep",
        "ready_s",
        "cmd",
        "env",
        "requested_spec_mode",
        "engaged_spec_mode",
        "engine_probe",
        "tag",
        "device",
        "contended",
        "product",
        "flags",
        "drafter",
        "spec_mode",
        "checkpoint",
        "weights_vs_oQ4e",
        "snapshot_rep",
    }
    end = {"job_end_s": 2.0, "gpu_exclusive": True, "contended": False, "swap_end": {}}
    have = set(fc) | emitted | set(end)
    required = set(fa.FULLCMP_SCHEMA["header"]) - {
        "contended"
    }  # contended is in emitted
    assert not (required - have), sorted(required - have)
    assert fc["model"] == "qwen-test"
    assert fc["cell"] == "cell"
    assert fc["release_date"] == "2026-11-03"
    assert fc["weights"] == [
        {"file": SHARD, "sha256": WEIGHT_SHA, "bits": 4, "group_size": 64}
    ]
    assert fc["python"] == layout.pin["python_executable"]
    assert fc["binary_sha256"] and len(fc["binary_sha256"]) == 64


def test_release_date_refuses_stale_census(layout):
    with pytest.raises(fp.ProvenanceError, match="TFB_RELEASE_DATE"):
        fp.release_date(layout.census, "yunshu", "0.9.9", {})
    assert (
        fp.release_date(
            layout.census, "yunshu", "0.1.7.dev0", {"TFB_RELEASE_DATE": "2026-10-01"}
        )
        == "2026-10-01"
    )


# ---------------------------------------------------------------- (b) telemetry
def test_send_marks_request_start_first_token_end(fake_http, fake_telemetry):
    sampler = FakeSampler(1, 2, "x")
    tfbench._ACTIVE_TELEMETRY = sampler
    try:
        r = tfbench.send("http://127.0.0.1:1", tfbench.req("M", "hello", 8))
    finally:
        tfbench._ACTIVE_TELEMETRY = None
    events = [ev for ev, _ in sampler.marks]
    assert events == ["request_start", "first_token", "end"]
    assert sampler.marks[1][1]["chunks"] == 1
    assert sampler.marks[2][1]["completion_tokens"] == r["ct"] == 3
    assert sampler.marks[0][1]["req"] == sampler.marks[2][1]["req"]


def test_required_sampler_failure_is_an_error_record(layout, fake_telemetry, tmp_path):
    FakeSampler.FAIL = True
    with pytest.raises(fp.ProvenanceError, match="telemetry sampler did not start"):
        tfbench.start_cell_telemetry(
            1,
            2,
            str(tmp_path / "e" / "c-r0.jsonl"),
            "yunshu",
            {"TFB_TELEMETRY_DIR": str(tmp_path / "root")},
            required=True,
        )


def test_optional_sampler_failure_continues(fake_telemetry, tmp_path):
    FakeSampler.FAIL = True
    got = tfbench.start_cell_telemetry(
        1,
        2,
        str(tmp_path / "c-r0.jsonl"),
        "yunshu",
        {"TFB_TELEMETRY_DIR": str(tmp_path / "root")},
        required=False,
    )
    assert got is None


def test_sampler_path_and_stop(fake_telemetry, tmp_path):
    path = tfbench.start_cell_telemetry(
        7,
        8,
        str(tmp_path / "yunshu" / "c-r1.jsonl"),
        "yunshu",
        {"TFB_TELEMETRY_DIR": str(tmp_path / "root")},
        required=True,
    )
    assert path == str(tmp_path / "root" / "yunshu" / "c-r1.telemetry.jsonl")
    sampler = FakeSampler.instances[-1]
    assert sampler.pid == 7 and sampler.port == 8
    tfbench.stop_cell_telemetry()
    assert sampler.stopped and tfbench._ACTIVE_TELEMETRY is None


# ---------------------------------------------------------------- (d) swap + whole cell
class _FakeProc:
    pid = 4242

    def poll(self):
        return None


class _FakeServer:
    def __init__(self, layout):
        self.proc = _FakeProc()
        self.port = 18991
        self.url = "http://127.0.0.1:18991"
        self.model = "qwen-test"
        self.cmd = [layout.pin["python_executable"], "-m", "yunshu_cli", "serve"]
        self.home = layout.tmp / "home"
        self.requested_spec_mode = "mtp"
        self.engaged_spec_mode = "mtp"
        self.engaged_log_line = "VLM batch runner: draft=mtp"
        self.pin = fp.require_yunshu_pin(layout.good)
        self.extra_env = {}
        self.launch = None
        self.probe = None
        self.mem = None
        self.ready_s = 1.0
        self.killed = False

    def verify_spec_mode(self):
        return None

    def probe_text(self):
        return None

    def kill(self):
        self.killed = True


@pytest.fixture
def cell_env(layout, fake_http, fake_telemetry, monkeypatch, tmp_path):
    env = dict(
        layout.good,
        TFB_FULLCMP="1",
        TFB_RUN_ID="run1",
        TFB_GROUP="g1",
        GPUQ_JOB_ID="job1",
        TFB_TELEMETRY_DIR=str(tmp_path / "root"),
    )
    for k in (
        "TFB_FULLCMP",
        "TFB_RUN_ID",
        "TFB_GROUP",
        "GPUQ_JOB_ID",
        "TFB_TELEMETRY_DIR",
        "TFB_YUNSHU_BIN",
        "TFB_YUNSHU_SRC",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(tfbench, "was_contended", lambda *a, **k: False)
    monkeypatch.setattr(
        tfbench.bench_engines,
        "engine_version",
        lambda eng, src="": ("0.1.7.dev0", PIN_SHA),
    )
    server = _FakeServer(layout)
    monkeypatch.setattr(tfbench, "start_server", lambda *a, **k: server)
    swaps = iter([{"used_mb": 10.0}, {"used_mb": 30.0}])
    monkeypatch.setattr(fp, "swap_snapshot", lambda: next(swaps))

    def part_decode(s, out, a):
        r = tfbench.send(s.url, tfbench.req(s.model, "hi", 3))
        r["text"] = r.pop("_text")
        tfbench.emit(
            out,
            part="decode",
            ctx=1024,
            kind="prose",
            phase="cold",
            prompt_file=tfbench.decode_prompt_file("prose", 1024, False),
            cell=tfbench.cell_name(a.out, a.rep),
            **r,
        )

    monkeypatch.setattr(tfbench, "part_decode", part_decode)
    out = tmp_path / "engines" / "yunshu" / "prose-1024-r0.jsonl"
    out.parent.mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tfbench.py",
            "--engine",
            "yunshu",
            "--part",
            "decode",
            "--rep",
            "0",
            "--out",
            str(out),
            "--model",
            str(layout.model),
        ],
    )
    return types.SimpleNamespace(out=out, server=server, tmp=tmp_path)


def _rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def test_whole_cell_writes_complete_records(cell_env):
    tfbench.main()
    rows = _rows(cell_env.out)
    header = rows[0]
    assert header["part"] == "session"
    required = set(fa.FULLCMP_SCHEMA["header"])
    assert not (required - set(header)), sorted(required - set(header))
    # (d) swap at start and end, job window ordered, GPU exclusive
    assert header["swap_start"] == {"used_mb": 10.0}
    assert header["swap_end"] == {"used_mb": 30.0}
    assert header["job_start_s"] < header["job_end_s"]
    assert header["gpu_exclusive"] is True and header["contended"] is False
    assert header["telemetry_path"].endswith("prose-1024-r0.telemetry.jsonl")
    assert header["engaged_log_line"] == "VLM batch runner: draft=mtp"
    decode = next(r for r in rows if r.get("part") == "decode")
    required_req = set(fa.FULLCMP_SCHEMA["request"]) - {"error"}
    assert not (required_req - set(decode)), sorted(required_req - set(decode))
    assert any(r.get("part") == "part_done" for r in rows)
    assert not any(r.get("part") == "error" for r in rows)
    # the sampler ran beside the server pid and was stopped at the end of the cell
    sampler = FakeSampler.instances[-1]
    assert sampler.pid == 4242 and sampler.stopped
    assert [ev for ev, _ in sampler.marks] == [
        "request_start",
        "first_token",
        "end",
    ] * 3
    assert cell_env.server.killed


def test_sampler_failure_writes_error_record_and_no_result(cell_env):
    FakeSampler.FAIL = True
    with pytest.raises(fp.ProvenanceError):
        tfbench.main()
    rows = _rows(cell_env.out)
    errors = [r for r in rows if r.get("part") == "error"]
    assert errors and "telemetry sampler did not start" in errors[0]["error"]
    assert not any(r.get("part") == "part_done" for r in rows)
    assert not any(r.get("part") == "decode" for r in rows)
    assert cell_env.server.killed


def test_missing_identifiers_refuse_before_launch(cell_env, monkeypatch):
    monkeypatch.delenv("GPUQ_JOB_ID")
    monkeypatch.setattr(
        tfbench,
        "start_server",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("launched")),
    )
    with pytest.raises(fp.ProvenanceError, match="GPUQ_JOB_ID"):
        tfbench.main()


def test_engaged_line_parse_is_the_mode_line():
    log = "x\nVLM batch runner: spec=on draft=mtp block=6\ny\n"
    mode, line = tfbench.engaged_spec_mode_line("yunshu", log)
    assert mode == "mtp" and line == "VLM batch runner: spec=on draft=mtp block=6"
    assert tfbench.engaged_spec_mode_line("yunshu", "nothing") == (None, None)
    assert tfbench.engaged_spec_mode("yunshu", log) == "mtp"


def test_census_models_under_meta_are_found(tmp_path):
    """The lead's census.json keeps model entries under `_meta.models`; the lookup must find
    them (2026-11-03 smoke failed with 'not in census.json' before this was read)."""
    model = tmp_path / "Qwen3.8-27B-oQ4e-mtp"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"quantization": {"bits": 4, "group_size": 64}})
    )
    (model / SHARD).write_text("weights")
    census = {
        "_meta": {
            "models": {
                "qwen3.8-27b-4bit-jundot": {
                    "path": str(model),
                    "sha256": {SHARD: WEIGHT_SHA},
                }
            }
        },
        "yunshu": {"latest": "0.1.7.dev0 (repo pin)"},
    }
    key = fp.census_model_id(census, str(model))
    assert key == "qwen3.8-27b-4bit-jundot"
    rec = fp.weights_record(str(model), census, key)
    assert rec == [
        {"file": SHARD, "sha256": WEIGHT_SHA, "bits": 4, "group_size": 64}
    ]
    with pytest.raises(fp.ProvenanceError, match="not in census.json"):
        fp.census_model_id(census, str(tmp_path / "other"))


def test_census_named_registry_ids_have_a_census_key():
    """Every census-named engine id maps to its census entry, so the release date is looked up
    instead of failing after the server has already started (2026-11-03 smoke)."""
    census = {"tensorfold": {"latest": "1.2.1", "latest_date": "2026-11-01T23:52:33Z"}}
    key = tfbench.FULLCMP_CENSUS_KEY["tf-1.2.1"]
    assert key == "tensorfold"
    assert fp.release_date(census, key, "1.2.1", {}) == "2026-11-01"
    with pytest.raises(fp.ProvenanceError):
        fp.release_date(census, key, "1.1.8", {})  # not the latest: needs TFB_RELEASE_DATE
    assert fp.release_date(census, key, "1.1.8", {"TFB_RELEASE_DATE": "2026-01-01"}) == "2026-01-01"


def test_tensorfold_1_2_engaged_line_is_recorded():
    """TensorFold 1.2.x names its drafter on the serving line; the audit needs that line recorded
    (2026-11-03 smoke: engaged_log_line was None although the server ran DFlash2)."""
    log = (
        "[tensorfold] listening\n"
        "[tensorfold] serving Qwen3.8-27B-MLX-4bit at http://127.0.0.1:18990/v1 (drafts: DFlash2 (/m/d); context: 140000)\n"
    )
    mode, line = tfbench.engaged_spec_mode_line("tf-1.2.1", log)
    assert mode == "dflash" and "drafts: DFlash2" in line
    assert tfbench.engaged_spec_mode_line("omlx-0.7.3", log) == (None, None)


def test_missing_gpuq_job_id_fails_closed_with_a_clear_message():
    """gpuq exports GPUQ_JOB_ID to every job; a fullcmp cell started outside gpuq must refuse."""
    env = {"TFB_RUN_ID": "run1", "TFB_GROUP": "g1"}
    with pytest.raises(fp.ProvenanceError, match="GPUQ_JOB_ID"):
        fp.check_required_env(env)
    fp.check_required_env(dict(env, GPUQ_JOB_ID="1103-1-x"))
