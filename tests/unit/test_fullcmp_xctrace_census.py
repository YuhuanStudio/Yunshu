"""Op census and host-sync parsing from Metal System Trace tables (fullcmp_xctrace_census, fullcmp_xctrace).

CPU only. The synthetic tables follow the real export layout (schema columns, ref= back-references, process
elements carrying a pid). One test reads the real dev0 trace export of the e67748f3c validation when present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import fullcmp_attribution as fa  # noqa: E402
import fullcmp_profile as fp  # noqa: E402
import fullcmp_xctrace as xt  # noqa: E402
import fullcmp_xctrace_census as cen  # noqa: E402

PID = 4242
OTHER = 99


def _table(schema: str, cols: list[str], rows: list[list[str]]) -> str:
    """A trace-query-result export: schema columns, then one <row> per list, children in column order. A
    cell given as 'process:<pid>' / 'thread:<tid>' is an element with a pid / tid child (like the export)."""
    head = "".join(f"<col><mnemonic>{c}</mnemonic></col>" for c in cols)
    body = []
    for n, row in enumerate(rows):
        cells = []
        for k, v in enumerate(row):
            if v.startswith("process:"):
                pid = v.split(":")[1]
                cells.append(f'<process id="p{n}_{k}" fmt="python ({pid})"><pid fmt="{pid}">{pid}</pid></process>')
            elif v.startswith("thread:"):
                tid = v.split(":")[1]
                cells.append(f'<thread id="t{n}_{k}" fmt="mlx"><tid fmt="{tid}">{tid}</tid></thread>')
            else:
                cells.append(f"<x>{v}</x>")
        body.append("<row>" + "".join(cells) + "</row>")
    return f'<?xml version="1.0"?><trace-query-result><node><schema name="{schema}">{head}</schema>' + "".join(body) + "</node></trace-query-result>"


GPU_COLS = ["start", "duration", "channel-name", "process", "cmdbuffer-id"]
SUB_COLS = ["start", "duration", "num-encoders", "process", "thread", "cmdbuffer-id"]


def _gpu(rows):
    return _table("metal-gpu-intervals", GPU_COLS, [[str(s), str(d), ch, f"process:{pid}", cb] for s, d, ch, pid, cb in rows])


def _sub(rows):
    return _table(
        "metal-application-command-buffer-submissions",
        SUB_COLS,
        [[str(s), "10", str(n), f"process:{pid}", "thread:7", cb] for s, n, pid, cb in rows],
    )


def test_read_table_resolves_columns_and_pid():
    rows = cen.read_table(_gpu([(100, 50, "Compute", PID, "cb1")]))
    assert rows[0]["start"] == "100" and rows[0]["channel-name"] == "Compute"
    assert rows[0]["process.pid"] == str(PID)


def test_server_encoders_filters_pid_and_compute_channel_and_sorts():
    xml = _gpu(
        [
            (300, 10, "Compute", PID, "b"),
            (100, 10, "Compute", PID, "a"),
            (200, 10, "Compute", OTHER, "z"),
            (250, 10, "Vertex", PID, "v"),
        ]
    )
    enc = cen.server_encoders(cen.read_table(xml), PID)
    assert [e[2] for e in enc] == ["a", "b"]


def test_op_census_classes_counts_and_per_step():
    us = 1000  # ns per microsecond
    enc = [(i * 10**6, d * us, "c") for i, d in enumerate([5, 50, 300, 300, 1000, 5000])]
    out = cen.op_census(enc, steps=2)
    assert out["lt_20us"]["count"] == 1 and out["20_100us"]["count"] == 1
    assert out["100_500us"]["count"] == 2 and out["500us_2ms"]["count"] == 1 and out["ge_2ms"]["count"] == 1
    assert out["100_500us"]["count_per_step"] == 1.0
    assert out["ge_2ms"]["gpu_ms"] == 5.0
    assert cen.op_census(enc, steps=None)["lt_20us"]["count_per_step"] is None  # never guessed


def test_idle_gaps_and_host_syncs_ignore_overlap_and_small_gaps():
    # a: [0,100) ; b overlaps a (start 50) ; c starts 30 us after the latest end (100 ns*..) -> use ns
    enc = [(0, 100_000, "a"), (50_000, 100_000, "b"), (180_000, 10_000, "c"), (500_000, 10_000, "d")]
    gaps = cen.idle_gaps_us(enc)
    assert gaps == [30.0, 310.0]  # c starts 30 us after b ends; d starts 310 us after c ends
    assert cen.host_syncs(enc, threshold_us=50.0) == 1
    assert cen.host_syncs(enc, threshold_us=10.0) == 2


def test_decode_window_starts_after_the_longest_gap():
    enc = [(0, 1000, "p"), (10_000, 1000, "p"), (5_000_000, 1000, "d1"), (5_100_000, 1000, "d2")]
    assert cen.decode_window(enc) == (5_000_000, 5_100_000)
    assert cen.decode_window(enc[:1]) is None


def test_census_from_tables_reports_per_step_and_labels_the_proxy():
    gpu = _gpu(
        [(0, 1000, "Compute", PID, "p")]
        + [(10_000_000 + i * 1_000_000, 600_000, "Compute", PID, f"d{i}") for i in range(8)]
        + [(0, 5, "Compute", OTHER, "x")]
    )
    sub = _sub([(10_000_000, 1, PID, "d0"), (10_000_001, 0, PID, "d1"), (3, 1, OTHER, "x")])
    out = cen.census_from_tables(gpu, sub, PID, steps=4)
    assert out["trace"]["encoder_executions"] == 8 and out["trace"]["steps"] == 4
    assert out["op_census"]["500us_2ms"]["count"] == 8 and out["op_census"]["500us_2ms"]["count_per_step"] == 2.0
    assert out["host_sync_count"] == pytest.approx(7 / 4)  # seven 400 us gaps over four steps
    assert out["trace"]["empty_command_buffers"] == 1
    assert "proxy" in out["trace"]["note"] and "no kernel names" in out["trace"]["note"]
    assert cen.census_from_tables(gpu, sub, PID, steps=None)["host_sync_count"] is None


def test_steps_from_log_takes_the_largest_decode_step_count():
    text = (
        "round driver idle: 1 decode steps, 89.1 ms in a step, 0.0 ms between steps\n"
        "round driver idle: 115 decode steps, 48.4 ms in a step, 0.3 ms between steps\n"
        "DFlash round profile: 36 rounds, 12.9 nodes/round, draft 2.5 ms, verify 45.7 ms, commit+host 0.0 ms\n"
    )
    assert cen.steps_from_log(text) == 115
    assert cen.steps_from_log("nothing") is None


REAL = Path("/Volumes/Micron/yunshu-fullcmp/xctrace/inspect2")
REAL_LOG = Path("/Volumes/Micron/yunshu-fullcmp/xctrace/runs/125846/server.log")


def test_real_dev0_trace_export_gives_a_plausible_census():
    """The e67748f3c validation trace (cold 1K code, out 128, dev0, server pid 50342): 36 rounds, 4173 server
    encoder executions in the decode window, about five GPU idle gaps per round."""
    gpu, sub = REAL / "metal-gpu-intervals.xml", REAL / "metal-application-command-buffer-submissions.xml"
    if not (gpu.is_file() and sub.is_file() and REAL_LOG.is_file()):
        pytest.skip("real trace export not on this machine")
    steps = cen.steps_from_log(REAL_LOG.read_text(errors="replace"))
    assert steps == 36
    out = cen.census_from_tables(gpu.read_text(), sub.read_text(), 50342, steps)
    assert out["trace"]["encoder_executions"] == pytest.approx(4173, abs=60)
    assert 4.0 < out["host_sync_count"] < 6.0
    assert out["op_census"]["500us_2ms"]["count"] > 2000


# --- capture driver glue and the audit side -----------------------------------------------------------


def test_append_census_row_writes_a_part_xctrace_row_and_errors_fail_closed(tmp_path):
    f = tmp_path / "c-prof.jsonl"
    f.write_text('{"part": "profile"}\n')
    ok = xt.append_census_row(f, "yunshu-pb2", {"op_census": {"x": 1}, "host_sync_count": 4.5, "trace": {"steps": 3}})
    bad = xt.append_census_row(f, "yunshu-pb2", {"error": "export failed"})
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    assert rows[1]["part"] == "xctrace" and rows[1]["host_sync_count"] == 4.5 and ok["op_census"] == {"x": 1}
    assert rows[2]["error"] == "export failed" and "op_census" not in bad


def test_trace_census_exports_two_tables_and_needs_a_pid(tmp_path, monkeypatch):
    import types

    gpu = _gpu([(0, 1000, "Compute", PID, "p")] + [(10_000_000 + i * 900_000, 500_000, "Compute", PID, f"d{i}") for i in range(4)])
    sub = _sub([(10_000_000, 1, PID, "d0")])
    seen = []

    def fake_run(cmd, timeout):
        seen.append(cmd[-2] if cmd[-1] == "--quiet" else cmd)
        xml = gpu if "metal-gpu-intervals" in " ".join(cmd) else sub
        return types.SimpleNamespace(returncode=0, stdout=xml, stderr="")

    monkeypatch.setattr(xt, "_run", fake_run)
    out = xt.trace_census(tmp_path / "t.trace", PID, "round driver idle: 4 decode steps, 1.0 ms in a step, 0.0 ms between steps", "c")
    assert out["trace"]["steps"] == 4 and out["cell"] == "c" and len(seen) == 2
    with pytest.raises(RuntimeError, match="no server pid"):
        xt.trace_census(tmp_path / "t.trace", None, "", "c")


def _prof_row_without_counters():
    return {
        "part": "profile",
        "engine": "yunshu",
        "stages": {"decode_round": {"drafter": 1.0, "commit": 0.1}},
    }


def test_audit_reads_host_sync_and_census_from_the_xctrace_row():
    """Failing-before: the counters had only an 'unavailable' reason. With an xctrace row in the profile file
    the row carries the measured values; an xctrace row that recorded an error leaves them MISSING."""
    base = _prof_row_without_counters()
    xt_row = {"part": "xctrace", "op_census": {"500us_2ms": {"count": 3}}, "host_sync_count": 4.2, "trace": {"steps": 36}}
    merged = fp.merge_xctrace(base, [base, xt_row])
    merged["layer2_unavailable"] = fp.unavailable_layer2(merged)
    cov = fa.check_profile("yunshu", merged)
    assert cov["host_sync_count"] == "present" and cov["op_census"] == "present"
    assert "host_sync_count" not in merged["layer2_unavailable"]

    failed = fp.merge_xctrace(base, [base, {"part": "xctrace", "error": "export of x failed"}])
    failed["layer2_unavailable"] = fp.unavailable_layer2(failed)
    miss = fa.missing(fa.check_profile("yunshu", failed))
    assert "host_sync_count" in miss and "op_census" in miss  # fail closed

    plain = fp.merge_xctrace(base, [base])
    assert plain is base  # no xctrace row: untouched; the cell then carries the 'one cold 1K request' reason
    assert "cold 1K code request" in fp.unavailable_layer2(plain)["host_sync_count"]
