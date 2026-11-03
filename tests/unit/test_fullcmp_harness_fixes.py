"""Harness fixes: one prompt corpus for every cell, the prompt-file sha guard, one engine-log slice per cell.

Stub-level: no model, no GPU. The shared server log is a plain file here; the slice helpers are the
same functions tfbench and the audit use.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_audit as fa  # noqa: E402
import fullcmp_run as fr  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "tfbench", ROOT / "scripts/research/tfbench.py"
)
tfbench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tfbench)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---- 1. TFB_WORK in every cell ------------------------------------------------------------
PIN = {
    "python_executable": "/venv/bin/python",
    "launch": {
        "harness_env": {
            "TFB_YUNSHU_BIN": "/bin/yunshu",
            "TFB_YUNSHU_SRC": "/src",
            "PYTHONPATH": "/src/python",
        }
    },
}


def _row(**kw):
    base = {
        "id": "c1",
        "engine": "yunshu",
        "batch": "B1",
        "model": "m",
        "thinking": "off",
        "prompt_kind": "code",
        "ctx": 1024,
        "ctx_tokens": 1024,
        "conc": 1,
        "status": "unverified",
    }
    base.update(kw)
    return base


def test_cell_env_exports_tfb_work_for_every_cell():
    env = fr.cell_env("yunshu", PIN, label="x", group="g")
    assert f"TFB_WORK={fr.TFB_WORK_DIR}" in env
    assert fr.TFB_WORK_DIR == "/Volumes/P5Plus/yunshu-build/snapshot014-b2"


def test_profile_cell_command_also_exports_tfb_work(monkeypatch):
    monkeypatch.setattr(fr, "harness_part", lambda row: ("decode", [], None))
    cmd = fr.cell_command(_row(), 0, PIN, label="x", profile=True)
    assert cmd is not None
    assert f"TFB_WORK={fr.TFB_WORK_DIR}" in cmd
    cmd_timed = fr.cell_command(_row(), 1, PIN, label="x", profile=False)
    assert f"TFB_WORK={fr.TFB_WORK_DIR}" in cmd_timed


def _corpus(tmp_path, names):
    (tmp_path / "prompts").mkdir()
    for name in names:
        (tmp_path / "prompts" / name).write_text("x")
    return str(tmp_path)


def test_corpus_missing_dir_fails_closed(tmp_path):
    problems = fr.prompt_corpus_problems([_row()], str(tmp_path / "nope"))
    assert problems and "prompt dir missing" in problems[0]


def test_corpus_missing_prompt_file_fails_closed(tmp_path):
    work = _corpus(tmp_path, ["code-8192.txt"])
    problems = fr.prompt_corpus_problems([_row(ctx_tokens=1024)], work)
    assert problems == [f"missing prompt file {tmp_path / 'prompts' / 'code-1024.txt'}"]


def test_corpus_family_kind_reads_the_prose_file(tmp_path):
    work = _corpus(tmp_path, ["prose-4096.txt"])
    assert fr.prompt_corpus_problems(
        [_row(prompt_kind="tool-chat", ctx_tokens=4096)], work
    ) == []


def test_corpus_complete_passes_and_skips_rows_without_a_decode_prompt(tmp_path):
    work = _corpus(tmp_path, ["code-1024.txt"])
    rows = [_row(), {"id": "agent", "engine": "yunshu"}]
    assert fr.prompt_corpus_problems(rows, work) == []


# ---- 2. prompt_file_sha256 guard -----------------------------------------------------------
def _run_with(rows):
    cell = {"cell": "c1", "group": "g", "engine": "yunshu"}
    cf = fa.CellFile(cell, 0, Path("/x/yunshu/c1-r0.jsonl"), {"part": "session"}, rows)
    return fa.Run(manifest={}, files=[cf], missing=[], extra=[], parse_errors=[])


def _decode_row(name, sha):
    return {
        "part": "decode",
        "phase": "cold",
        "prompt_file": f"/Volumes/P5Plus/yunshu-build/snapshot014-b2/prompts/{name}",
        "prompt_file_sha256": sha,
        "prompt_ids_sha": "rendered",
        "pt": 1024,
    }


@pytest.fixture
def table(tmp_path, monkeypatch):
    body = f"{_sha('code-1024 bytes')}  code-1024.txt\n{_sha('code-8192 bytes')}  code-8192.txt\n"
    path = tmp_path / "prompt_files.sha256.txt"
    path.write_text(body)
    monkeypatch.setattr(fa, "PROMPT_SHA_FILE", path)
    return path


def test_prompt_identity_passes_when_file_sha_matches_table(table):
    run = _run_with([_decode_row("code-1024.txt", _sha("code-1024 bytes"))])
    res = fa.check_prompt_identity(run)
    assert res.violations == []


def test_prompt_identity_wrong_sha_fails(table):
    run = _run_with([_decode_row("code-1024.txt", _sha("other bytes"))])
    res = fa.check_prompt_identity(run)
    assert any("sha" in v and "!=" in v for v in res.violations)


def test_prompt_identity_missing_table_entry_fails(table):
    run = _run_with([_decode_row("code-4096.txt", _sha("code-4096 bytes"))])
    res = fa.check_prompt_identity(run)
    assert any("no entry in the sha table" in v for v in res.violations)


def test_prompt_identity_missing_row_sha_fails(table):
    row = _decode_row("code-1024.txt", None)
    res = fa.check_prompt_identity(_run_with([row]))
    assert any("not recorded" in v for v in res.violations)


def test_prompt_identity_unreadable_table_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(fa, "PROMPT_SHA_FILE", tmp_path / "absent.txt")
    res = fa.check_prompt_identity(_run_with([_decode_row("code-1024.txt", _sha("a"))]))
    assert res.violations and "unreadable" in res.violations[0]


def test_prompt_file_sha_is_the_file_bytes(tmp_path, monkeypatch):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "code-1024.txt").write_bytes(b"corpus text")
    monkeypatch.setattr(tfbench, "WORK", tmp_path)
    assert tfbench.prompt_file_sha256("code", 1024) == _sha("corpus text")
    assert tfbench.prompt_file_sha256("code", 1024, smoke=True) is None


# ---- 3. one engine-log slice per cell ------------------------------------------------------
def _profile_line(engine, n):
    return f"[round-profile] engine={engine} round={n} drafts=4 accepted=3 tokens={n}\n"


def test_server_log_path_sits_beside_the_cell_file():
    out = "/raw/yunshu/c1-r2.jsonl"
    assert tfbench.server_log_path(out) == "/raw/yunshu/c1-r2.server.log"
    assert tfbench.server_log_path("/p/yunshu/c1-prof.jsonl") == "/p/yunshu/c1-prof.server.log"
    assert str(fa.server_log_slice_path(out)) == tfbench.server_log_path(out)


def test_two_cells_in_one_log_get_two_distinct_slices(tmp_path):
    """Stub chain: cell A and cell B append to one shared engine log; each slice keeps its own lines."""
    log = tmp_path / "server-yunshu-decode-0.log"
    log.write_bytes(b"[engine] ready\n")
    a_start = log.stat().st_size
    with open(log, "ab") as fh:
        fh.write("".join(_profile_line("yunshu", n) for n in (1, 2)).encode())
    a_end = log.stat().st_size
    b_start = a_end
    with open(log, "ab") as fh:
        fh.write("".join(_profile_line("yunshu", n) for n in (7, 8, 9)).encode())
    b_end = log.stat().st_size
    slice_a = tfbench.cell_log_slice(log, a_start, a_end)
    slice_b = tfbench.cell_log_slice(log, b_start, b_end)
    assert slice_a != slice_b
    assert slice_a.count(b"[round-profile]") == 2 and b"round=1 " in slice_a
    assert slice_b.count(b"[round-profile]") == 3 and b"round=7 " in slice_b
    assert b"round=7 " not in slice_a and b"round=1 " not in slice_b
    assert b"[engine] ready" not in slice_a  # startup lines are outside the cell's range


def test_cell_slice_survives_a_later_cell_overwriting_the_log(tmp_path):
    """The shared log is truncated by the next cell's launch: the slice written at cell end stays."""
    log = tmp_path / "server.log"
    log.write_bytes(_profile_line("yunshu", 1).encode())
    slice_path = tmp_path / "c1-r0.server.log"
    slice_path.write_bytes(tfbench.cell_log_slice(log, 0, log.stat().st_size))
    log.write_bytes(_profile_line("yunshu", 99).encode())  # the next cell's launch
    assert b"round=1 " in slice_path.read_bytes()


def test_restart_prefix_keeps_the_bytes_the_relaunch_truncated(tmp_path):
    """A restart cell: the first engine's log is replaced by the relaunch's; the slice joins both."""
    log = tmp_path / "server.log"
    log.write_bytes(b"[engine] ready\n" + _profile_line("yunshu", 1).encode())
    start = len(b"[engine] ready\n")
    before = log.read_bytes()
    log.write_bytes(_profile_line("yunshu", 2).encode())  # relaunch truncated the file
    data = tfbench.cell_log_slice(log, start, None, prefix=[before])
    assert b"round=1 " in data and b"round=2 " in data
    assert b"[engine] ready" not in data


@pytest.mark.parametrize("content", [None, b""])
def test_slice_gate_fails_closed_when_missing_or_empty(tmp_path, content):
    cell = tmp_path / "yunshu" / "c1-r0.jsonl"
    cell.parent.mkdir(parents=True)
    cell.write_text("")
    slice_path = fa.server_log_slice_path(cell)
    if content is not None:
        slice_path.write_bytes(content)
    problem = fa.server_log_slice_problem(cell)
    assert problem is not None
    assert ("missing" in problem) if content is None else ("empty" in problem)


def test_slice_gate_passes_when_present_and_non_empty(tmp_path):
    cell = tmp_path / "yunshu" / "c1-r0.jsonl"
    cell.parent.mkdir(parents=True)
    cell.write_text("")
    fa.server_log_slice_path(cell).write_bytes(_profile_line("yunshu", 1).encode())
    assert fa.server_log_slice_problem(cell) is None
