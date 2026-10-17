"""Published benchmark copies must preserve the measured rows and failure status."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def table_rows(text):
    return [line for line in text.splitlines() if line.startswith("| ")]


def test_console_benchmark_tables_preserve_public_evidence_rows():
    evidence = (ROOT / "docs/reports/RELEASE_015_BENCHMARKS.md").read_text()
    rows = table_rows(evidence)
    assert len(rows) == 68
    for locale in ("", ".zh-TW", ".zh-CN"):
        page = (ROOT / f"frontend/docs/guides/benchmarks{locale}.mdx").read_text()
        assert all(row in page for row in rows)
        assert "no part_done (incomplete job)" in page
        assert "unusable: garbage output" in page
        assert "overall verdict was FAIL" in page


def test_release_comparison_preserves_weight_and_sample_limits():
    text = (ROOT / "docs/BENCHMARKS.md").read_text()
    for required in (
        "134.400",
        "117.550",
        "77.950",
        "86.400",
        "98.700",
        "two valid",
        "reused competitors one",
        "different weights",
        "128K is unmeasured",
        "Flash-Next follow-up",
    ):
        assert required in text
