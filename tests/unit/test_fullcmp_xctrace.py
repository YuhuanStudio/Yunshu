"""CPU-only tests for the xctrace capture driver (scripts/research/fullcmp_xctrace.py).

No xctrace run, no server: trace layout, record/export command construction, toc schema
selection, row/duration parsing and the busy-percent arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_xctrace as xc  # noqa: E402


def test_trace_layout_is_under_micron_by_engine_and_cell():
    p = xc.trace_path(
        "yunshu-main",
        "code-1k-out128",
        root=Path("/Volumes/Micron/yunshu-fullcmp/xctrace"),
    )
    assert p == Path(
        "/Volumes/Micron/yunshu-fullcmp/xctrace/yunshu-main/code-1k-out128.trace"
    )


def test_record_command_attaches_metal_template_with_no_prompt():
    cmd = xc.record_cmd(4242, Path("/x/y.trace"), 120)
    assert cmd[0] == xc.XCTRACE
    assert cmd[1:3] == ["record", "--template"]
    assert cmd[3] == "Metal System Trace"
    assert cmd[cmd.index("--attach") + 1] == "4242"
    assert cmd[cmd.index("--output") + 1] == "/x/y.trace"
    assert cmd[cmd.index("--time-limit") + 1] == "120s"
    assert "--no-prompt" in cmd
    assert "--launch" not in cmd  # attach mode only


def test_export_commands():
    assert xc.export_toc_cmd(Path("/x.trace"))[1:4] == ["export", "--input", "/x.trace"]
    cmd = xc.export_table_cmd(Path("/x.trace"), "metal-gpu-intervals")
    xpath = cmd[cmd.index("--xpath") + 1]
    assert 'table[@schema="metal-gpu-intervals"]' in xpath
    assert xpath.startswith('/trace-toc/run[@number="1"]/data')


def test_gpu_schema_selection_from_toc():
    toc = (
        '<toc><run number="1"><data>'
        '<table schema="metal-gpu-intervals"/>'
        '<table schema="metal-command-buffer-completed"/>'
        '<table schema="cpu-profile"/>'
        '<table schema="metal-gpu-intervals"/>'
        "</data></run></toc>"
    )
    assert xc.gpu_schemas(toc) == [
        "metal-gpu-intervals",
        "metal-command-buffer-completed",
    ]
    assert xc.gpu_schemas("<toc/>") == []


def test_union_counts_overlaps_once():
    # [0,10) and [5,15) cover 15; [20,25) adds 5
    assert xc.union_ns([(0, 10), (5, 10), (20, 5)]) == 20
    assert xc.union_ns([]) == 0


def test_gpu_summary_resolves_refs_and_filters_by_pid():
    table = (
        '<trace-query-result><node><schema name="metal-gpu-intervals"/>'
        '<row><start-time id="1">100</start-time><duration id="2">50</duration>'
        '<process id="3" fmt="python (4242)"/><gpu-channel-name id="4" fmt="Compute">Compute</gpu-channel-name></row>'
        "<row><start-time>120</start-time><duration>50</duration>"
        '<process ref="3"/><gpu-channel-name ref="4"/></row>'
        "<row><start-time>400</start-time><duration>100</duration>"
        '<process id="9" fmt="WindowServer (634)"/><gpu-channel-name>Vertex</gpu-channel-name></row>'
        "</node></trace-query-result>"
    )
    s = xc.gpu_summary(table, pid=4242)
    assert s["gpu_rows"] == 3
    assert s["gpu_rows_server"] == 2
    assert s["gpu_busy_ns_server"] == 70  # [100,150) U [120,170)
    assert s["gpu_span_ns_server"] == 70
    assert s["gpu_busy_ns_all"] == 170  # [100,170) and [400,500)
    assert s["gpu_span_ns_all"] == 400  # 100 .. 500
    assert xc.gpu_summary("<trace-query-result/>", pid=1)["gpu_rows"] == 0


def test_busy_pct_clamps_and_fails_closed_on_empty_window():
    assert xc.busy_pct(500, 1000) == 50.0
    assert xc.busy_pct(2000, 1000) == 100.0  # overlap cannot exceed the window
    assert xc.busy_pct(10, 0) is None


def test_ioreg_util_parse():
    assert xc.ioreg_util('"Device Utilization %"=37,"x"=1') == 37
    assert xc.ioreg_util("nothing here") is None
