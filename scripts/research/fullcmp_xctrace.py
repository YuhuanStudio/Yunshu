#!/usr/bin/env python3
"""Metal System Trace capture of one cold 1K code request (profile-rep only, never a table number).

Flow (one server, one cell): start the pinned Yunshu server (fullcmp_sampler_ab launch: same
TFB env and pin), wait ready, attach `xctrace record --template 'Metal System Trace' --attach <pid>`,
send the cold 1K code request (out 128) inside the attach window, stop xctrace (SIGINT, so it
finalizes the .trace), then send one untraced request with the same prompt as the decode baseline.
The baseline is warm-prefix (the cold one already populated the cache); decode tok/s is compared,
not prefill. Then validate the trace: `export --toc` and the Metal GPU interval table.

Pure helpers (trace layout, command construction, toc/row parsing) carry the CPU unit test.
Run only through gpuq: scripts/dev/gpuq run --priority 1 --short --serving-ok -- <this script>.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

XCTRACE = "/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace"
TEMPLATE = "Metal System Trace"
MICRON_ROOT = Path("/Volumes/Micron/yunshu-fullcmp")
XCTRACE_DIR = MICRON_ROOT / "xctrace"
TMP_DIR = MICRON_ROOT / "tmp"
CELL = "code-1k-out128"
ENGINE = "yunshu-main"
OUT_TOKENS = 128
ATTACH_SETTLE_S = 8.0
PORT = 18992  # 18990-18998 are the shared band; 18991 belongs to the sampler A/B driver


def trace_path(
    engine: str = ENGINE, cell: str = CELL, root: Path = XCTRACE_DIR
) -> Path:
    """Where the .trace lands: <root>/<engine>/<cell>.trace."""
    return Path(root) / engine / f"{cell}.trace"


def record_cmd(pid: int, trace: Path, time_limit_s: int = 120) -> list[str]:
    """xctrace attach-mode recording of Metal System Trace for one pid."""
    return [
        XCTRACE,
        "record",
        "--template",
        TEMPLATE,
        "--attach",
        str(pid),
        "--output",
        str(trace),
        "--time-limit",
        f"{time_limit_s}s",
        "--no-prompt",
        "--quiet",
    ]


def export_toc_cmd(trace: Path) -> list[str]:
    return [XCTRACE, "export", "--input", str(trace), "--toc", "--quiet"]


def export_table_cmd(trace: Path, schema: str) -> list[str]:
    xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    return [XCTRACE, "export", "--input", str(trace), "--xpath", xpath, "--quiet"]


def gpu_schemas(toc_xml: str) -> list[str]:
    """Table schema names in a toc export that look like Metal GPU interval tables."""
    names = re.findall(r'<table[^>]*\bschema="([^"]+)"', toc_xml)
    out = []
    for n in dict.fromkeys(names):
        low = n.lower()
        if "metal" in low and ("gpu" in low or "interval" in low or "command" in low):
            out.append(n)
    return out


def count_rows(table_xml: str) -> int:
    """Number of data rows in an exported table (each <row> element is one event)."""
    return len(re.findall(r"<row[ >/]", table_xml))


def busy_pct(busy_ns: int, window_ns: int) -> float | None:
    if window_ns <= 0:
        return None
    return round(100.0 * min(busy_ns, window_ns) / window_ns, 2)


def ioreg_util(text: str) -> int | None:
    m = re.search(r'"Device Utilization %"=(\d+)', text)
    return int(m.group(1)) if m else None


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    env = dict(os.environ, TMPDIR=str(TMP_DIR))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)


def _ioreg_sampler(stop: threading.Event, out: list) -> None:
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-c", "AGXAccelerator"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            v = ioreg_util(r.stdout)
            if v is not None:
                out.append((time.time(), v))
        except Exception:  # noqa: BLE001 - a missed sample is not fatal
            pass
        stop.wait(0.5)


GPU_TABLE = "metal-gpu-intervals"


def _pool(root) -> dict:
    """id -> element, for resolving the export's ref= back-references."""
    return {el.get("id"): el for el in root.iter() if el.get("id") is not None}


def _value(el, pool: dict):
    if el is None:
        return None
    if el.get("ref") is not None:
        el = pool.get(el.get("ref"))
        if el is None:
            return None
    text = (el.text or "").strip()
    return text if text else el.get("fmt")


def gpu_intervals(table_xml: str) -> list[tuple[int, int, str, str]]:
    """(start_ns, duration_ns, process label, channel) for every metal-gpu-intervals row."""
    root = ET.fromstring(table_xml)
    pool = _pool(root)
    out = []
    for row in root.iter("row"):
        start = _value(row.find("start-time"), pool)
        dur = _value(row.find("duration"), pool)
        if not start or not dur:
            continue
        proc = _value(row.find("process"), pool) or ""
        chan = _value(row.find("gpu-channel-name"), pool) or ""
        out.append((int(start), int(dur), str(proc), str(chan)))
    return out


def union_ns(intervals: list[tuple[int, int]]) -> int:
    """Time covered by the union of [start, start+dur) intervals (overlaps counted once)."""
    total = 0
    cur_s = cur_e = None
    for s, d in sorted(intervals):
        e = s + d
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def gpu_summary(table_xml: str, pid: int | None) -> dict:
    """Busy % from the Metal GPU interval table for the server pid and for all GPU work.

    The denominator is the span from the first to the last interval of each set: the trace
    clock carries no request marker, so the wall-clock request window is reported beside it.
    """
    rows = gpu_intervals(table_xml)
    mine = [(s, d) for s, d, proc, _ in rows if pid is not None and f"({pid})" in proc]
    everything = [(s, d) for s, d, _, _ in rows]

    out: dict = {"gpu_rows": len(rows), "gpu_rows_server": len(mine)}
    for name, iv in (("server", mine), ("all", everything)):
        if iv:
            a = min(s for s, _ in iv)
            b = max(s + d for s, d in iv)
        else:
            a = b = 0
        busy = union_ns(iv)
        out[f"gpu_busy_ns_{name}"] = busy
        out[f"gpu_span_ns_{name}"] = b - a
        out[f"gpu_busy_pct_{name}_span"] = busy_pct(busy, b - a)
    return out


def validate_gpu(trace: Path, pid: int | None) -> dict:
    """Export the metal-gpu-intervals table of a finished trace and summarize it."""
    exp = _run(export_table_cmd(trace, GPU_TABLE), 900)
    if exp.returncode != 0 or not exp.stdout.strip():
        return {"table_schema": GPU_TABLE, "table_rc": exp.returncode, "gpu_rows": 0}
    return {"table_schema": GPU_TABLE, "table_rc": 0, **gpu_summary(exp.stdout, pid)}


def yunshu_launch_paths(engine: str, ab) -> tuple[str, str, str]:
    """(binary, source dir, tree dir) of the Yunshu tree an engine traces: dev0 from the sampler A/B pins,
    the Phase B preview from its own pin json (harness_env + worktree)."""
    if engine == "yunshu-pb2":
        import fullcmp_provenance as fp

        pin = json.loads(fp.pin_path_for("yunshu-pb2").read_text())
        env = pin["launch"]["harness_env"]
        return env["TFB_YUNSHU_BIN"], env["TFB_YUNSHU_SRC"], pin["worktree"]
    return ab.PIN_YUNSHU_BIN, ab.PIN_YUNSHU_SRC, ab.TREE


def cold_cell_stem(a) -> str | None:
    """The profile cell stem the census row belongs to (from --profile-file), for the row's cell field."""
    return a.profile_file.stem if a.profile_file is not None else None


CENSUS_TABLES = (GPU_TABLE, "metal-application-command-buffer-submissions")


def trace_census(trace: Path, pid: int | None, server_log: str, cell: str | None) -> dict:
    """Export the two tables of a finished trace and compute the census fields (fullcmp_xctrace_census).
    The decode step count of the traced request comes from the engine log; without a pid or a step count
    the per-step numbers stay None (and the row says so)."""
    import fullcmp_xctrace_census as cen

    if pid is None:
        raise RuntimeError("no server pid recorded: the census needs the process to filter the trace")
    xml = {}
    for schema in CENSUS_TABLES:
        exp = _run(export_table_cmd(trace, schema), 900)
        if exp.returncode != 0 or not exp.stdout.strip():
            raise RuntimeError(f"export of {schema} failed (rc {exp.returncode})")
        xml[schema] = exp.stdout
    steps = cen.steps_from_log(server_log)
    out = cen.census_from_tables(xml[CENSUS_TABLES[0]], xml[CENSUS_TABLES[1]], pid, steps)
    out["cell"] = cell
    return out


def append_census_row(profile_file: Path, engine: str, census: dict) -> dict:
    """Append one part=xctrace row to the profile cell file; the audit reads host_sync_count / op_census
    from it (fullcmp_profile.merge_xctrace). An error census is appended as an error row (fail closed)."""
    row = {"part": "xctrace", "engine": engine, "template": TEMPLATE, "cell_traced": CELL}
    if "error" in census:
        row["error"] = census["error"]
    else:
        row.update(
            op_census=census["op_census"],
            host_sync_count=census["host_sync_count"],
            trace=census["trace"],
        )
    with Path(profile_file).open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--model-dir-check", action="store_true", help="only print the plan"
    )
    ap.add_argument(
        "--validate-only",
        type=Path,
        default=None,
        help="validate an existing .trace (no server, no GPU work)",
    )
    ap.add_argument("--pid", type=int, default=None, help="server pid for --validate-only")
    ap.add_argument(
        "--engine",
        choices=("yunshu-main", "yunshu-pb2"),
        default=ENGINE,
        help="harness id of the Yunshu tree to trace (its own pin supplies binary, source and tree)",
    )
    ap.add_argument(
        "--profile-file",
        type=Path,
        default=None,
        help="profile cell file (<profile root>/<engine>/<cell>-prof.jsonl): the census row is appended here",
    )
    a = ap.parse_args()
    if a.validate_only is not None:
        toc = _run(export_toc_cmd(a.validate_only), 300)
        res = {
            "trace": str(a.validate_only),
            "toc_rc": toc.returncode,
            "gpu_table_listed": GPU_TABLE in gpu_schemas(toc.stdout),
        }
        res.update(validate_gpu(a.validate_only, a.pid))
        print(json.dumps(res, indent=1))
        return 0 if res.get("gpu_rows_server", res.get("gpu_rows", 0)) > 0 else 1
    engine = a.engine
    trace = trace_path(engine)
    print(
        json.dumps(
            {
                "plan": "xctrace-1k-code",
                "trace": str(trace),
                "template": TEMPLATE,
                "out_tokens": OUT_TOKENS,
            }
        )
    )
    if a.model_dir_check:
        return 0

    import bench_engines as be  # noqa: E402
    import fullcmp_sampler_ab as ab  # noqa: E402 - pinned launch + readiness helpers
    import tfbench as tb  # noqa: E402

    for d in (trace.parent, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if trace.exists():
        raise RuntimeError(f"refusing to overwrite existing trace {trace}")
    if ab.port_open(PORT):
        raise RuntimeError(f"port {PORT} already in use; refusing to launch")

    run_dir = MICRON_ROOT / "xctrace" / "runs" / time.strftime("%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    home = run_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    bin_, src, tree = yunshu_launch_paths(engine, ab)
    environ = dict(
        os.environ,
        TFB_YUNSHU_BIN=os.environ.get("TFB_YUNSHU_BIN", bin_),
        TFB_YUNSHU_SRC=os.environ.get("TFB_YUNSHU_SRC", src),
        TMPDIR=str(TMP_DIR),
    )
    launch = be.build_launch(engine, PORT, home, environ)
    for path, text in launch.files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    for link, target in launch.links.items():
        Path(link).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.lexists(link):
            os.symlink(target, link)

    url = f"http://127.0.0.1:{PORT}"
    log = run_dir / "server.log"
    rec: dict = {
        "engine": engine,
        "cell": CELL,
        "trace": str(trace),
        "run_dir": str(run_dir),
    }
    proc = None
    xct = None
    try:
        with open(log, "wb") as fh:
            proc = subprocess.Popen(
                launch.cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=launch.env,
                cwd=tree,
            )
        model = ab.wait_ready(proc, url, log)
        rec["model_id"] = model
        rec["server_pid"] = proc.pid
        xct_log = run_dir / "xctrace-record.log"
        with open(xct_log, "wb") as xfh:
            xct = subprocess.Popen(
                record_cmd(proc.pid, trace, 120),
                stdout=xfh,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, TMPDIR=str(TMP_DIR)),
            )
        time.sleep(
            ATTACH_SETTLE_S
        )  # attach + recorder warm-up before the request window
        if xct.poll() is not None:
            raise RuntimeError(
                f"xctrace exited before the request (rc={xct.returncode}); see {xct_log}"
            )

        text = tb.decode_prompt("code", 1024, True)
        ioreg: list = []
        stop = threading.Event()
        th = threading.Thread(target=_ioreg_sampler, args=(stop, ioreg), daemon=True)
        t0 = time.time()
        th.start()
        cold = tb.send(url, tb.req(model, text, OUT_TOKENS))
        t1 = time.time()
        tb.check_decode_len(cold, OUT_TOKENS, "traced cold 1K code")
        stop.set()
        th.join(timeout=5)
        rec.update(
            traced_dec_tps=cold.get("dec_tps"),
            traced_ttft_s=cold.get("ttft_s"),
            traced_window_s=round(t1 - t0, 3),
            ioreg_util_pct=[v for _, v in ioreg],
        )
        # stop recording: SIGINT lets xctrace finalize the bundle
        xct.send_signal(signal.SIGINT)
        try:
            xct.wait(timeout=300)
        except subprocess.TimeoutExpired:
            xct.kill()
            xct.wait(timeout=30)
            raise RuntimeError(
                "xctrace did not finalize within 300 s after SIGINT"
            ) from None
        rec["xctrace_rc"] = xct.returncode

        # untraced decode baseline: same prompt, prefix now warm (decode-equivalent comparison)
        base_body = tb.req(model, text, OUT_TOKENS)
        base_body["ignore_eos"] = True
        base = tb.send(url, base_body)
        tb.check_decode_len(base, OUT_TOKENS, "untraced baseline 1K code")
        rec.update(
            untraced_dec_tps=base.get("dec_tps"), untraced_ttft_s=base.get("ttft_s")
        )
    finally:
        if xct is not None and xct.poll() is None:
            xct.kill()
            xct.wait(timeout=30)
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=30)
        rec["server_rc"] = proc.returncode if proc is not None else None

    # validation: size, toc, GPU interval table
    rec["trace_bytes"] = (
        sum(p.stat().st_size for p in trace.rglob("*") if p.is_file())
        if trace.exists()
        else 0
    )
    toc = _run(export_toc_cmd(trace), 300)
    rec["toc_rc"] = toc.returncode
    schemas = gpu_schemas(toc.stdout) if toc.returncode == 0 else []
    rec["gpu_schemas"] = schemas
    if GPU_TABLE in schemas:
        rec.update(validate_gpu(trace, rec.get("server_pid")))
    try:
        census = trace_census(trace, rec.get("server_pid"), log.read_text(errors="replace"), cold_cell_stem(a))
    except Exception as exc:  # noqa: BLE001 - recorded, never silent: the profile row then says why
        census = {"error": f"{type(exc).__name__}: {exc}"}
    rec["census"] = census
    if a.profile_file is not None:
        append_census_row(a.profile_file, engine, census)
    t = rec.get("traced_dec_tps")
    b = rec.get("untraced_dec_tps")
    if t and b:
        rec["decode_overhead_pct"] = round(100.0 * (b - t) / b, 2)
    out = run_dir / "result.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    ok = toc.returncode == 0 and GPU_TABLE in schemas and rec.get("gpu_rows_server", 0) > 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
