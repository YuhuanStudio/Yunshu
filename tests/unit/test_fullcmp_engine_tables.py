"""Per-engine table registry (scripts/research/fullcmp_engine_tables.py).

CPU only: resolves the real census, yunshu_pin.json and engaged evidence table through the accessors
the harness uses at run time. No GPU, no gpuq, no server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_attribution as fa  # noqa: E402
import fullcmp_audit_manifest as fam  # noqa: E402
import fullcmp_engine_tables as fet  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402
import fullcmp_run as frun  # noqa: E402


def test_every_manifest_engine_resolves_every_table():
    """The GPU must never see a table miss. Fails while an engine's spec mode is 'unverified'
    (its engaged-mode guard fails every cell): that is an open decision, listed by this test."""
    misses = fet.check_all()
    assert misses == [], fet.format_misses(misses)


def test_registry_covers_manifest_ids_not_bench_ids():
    # the checked list is the manifest ids minus the stubs and the placeholder
    assert set(fet.CHECKED_IDS) == set(fm.ENGINES) - {"ollama", "lmstudio", fm.PLACEHOLDER}
    assert "yunshu" in fet.CHECKED_IDS and "yunshu-main" not in fet.CHECKED_IDS


@pytest.mark.parametrize(
    "engine", ["mlx-serve-26.11.1", "kapitsa-0.2.26", "llama-cpp-b12061", "yunshu"]
)
def test_attribution_schema_is_keyed_by_manifest_id(engine):
    """Failing before the fix: the schema was keyed 'mlx-serve', 'kapitsa', 'llama-cpp', so the
    tfbench lookup (bench_engines.manifest_id) raised KeyError for these engines."""
    assert fa.engine_attr(engine).engine == engine


def test_tensorfold_121_has_expected_mode_probe():
    """Failing before the fix: tf-1.2.1 had no EXPECTED_MODE_PROBE entry, so the audit silently
    defaulted its expected spec mode to 'unverified'."""
    probe = fam.EXPECTED_MODE_PROBE["tf-1.2.1"]
    assert probe["expected_mode"] == "dflash"
    assert any(p.get("confidence") == "high" for p in probe["log_patterns"])
    assert fam._expected("tf-1.2.1") == ("dflash", r"drafts: DFlash2\b")


def test_unknown_engine_id_is_a_miss():
    misses = fet.check_all(["no-such-engine"])
    assert [(m.engine, m.table) for m in misses] == [("no-such-engine", "manifest entry")]


def test_stub_and_placeholder_ids_are_skipped():
    assert fet.check_all(["ollama", "lmstudio", fm.PLACEHOLDER]) == []


def test_check_all_returns_every_miss_at_once(monkeypatch):
    """A table that fails for two engines reports both, with the key it tried and the error."""

    def broken(engine, ctx):
        if engine in ("tf-1.2.1", "yunshu"):
            raise fet.TableMiss(f"key-{engine}", "injected miss")
        return "ok"

    monkeypatch.setattr(fet, "TABLES", [(fet.Table("probe", "test", "manifest id"), broken)])
    misses = fet.check_all(["tf-1.2.1", "yunshu", "tf-1.1.8"])
    assert sorted((m.engine, m.table, m.key, m.error) for m in misses) == [
        ("tf-1.2.1", "probe", "key-tf-1.2.1", "injected miss"),
        ("yunshu", "probe", "key-yunshu", "injected miss"),
    ]


def test_unresolved_is_a_note_not_a_miss(monkeypatch):
    def documented(engine, ctx):
        raise fet.Unresolved("open item with a recorded reason")

    monkeypatch.setattr(fet, "TABLES", [(fet.Table("probe", "test", "manifest id"), documented)])
    misses, notes = fet.run_checks(["tf-1.2.1"])
    assert misses == [] and [n.text for n in notes] == ["open item with a recorded reason"]


@pytest.mark.parametrize("flag", ["--dry-run", "--self-check"])
def test_runner_refuses_on_a_miss_before_any_job(flag, monkeypatch, tmp_path, capsys):
    """--dry-run and --self-check (and a real submission) exit nonzero and list every miss; nothing
    is planned, queued or run."""
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text('{"batch": "B1", "engine": "yunshu", "id": "x"}\n')
    miss = fet.Miss("yunshu", "probe", "k", "injected")
    seen = []
    monkeypatch.setattr(fet, "check_all", lambda ids, ctx=None: seen.append(list(ids)) or [miss])
    monkeypatch.setattr(
        frun, "plan_batch", lambda *a, **k: pytest.fail("planned after a table miss")
    )
    monkeypatch.setattr(
        frun, "self_check", lambda *a, **k: pytest.fail("self-check ran after a table miss")
    )
    rc = frun.main(["--batch", "B1", "--manifest", str(manifest), "--engines", "yunshu", flag])
    err = capsys.readouterr().err
    assert rc == 2
    assert seen == [["yunshu"]]
    assert "yunshu: [probe] key 'k': injected" in err
