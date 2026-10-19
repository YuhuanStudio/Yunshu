"""Published benchmark copies preserve qualified evidence and failed coverage."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "docs/reports/RELEASE_016_BENCHMARKS_2026-10-19.md"


def table_rows(text):
    return [line for line in text.splitlines() if line.startswith("| ")]


def test_console_benchmark_tables_preserve_public_evidence_rows():
    evidence = SNAPSHOT.read_text()
    rows = table_rows(evidence)
    assert len([r for r in rows if "| 3/3 |" in r]) == 27
    for locale in ("", ".zh-TW", ".zh-CN"):
        page = (ROOT / f"frontend/docs/guides/benchmarks{locale}.mdx").read_text()
        assert all(row in page for row in rows)
        for status in (
            "output mismatch",
            "122.724 GiB",
            "concatenate shapes",
            "no speed ranking",
            "screenshots remain pending",
        ):
            assert status in page
    history = (ROOT / "docs/reports/RELEASE_015_BENCHMARKS.md").read_text()
    assert len(table_rows(history)) == 68
    assert "no part_done (incomplete job)" in history
    assert "unusable: garbage output" in history
    assert "overall verdict was FAIL" in history


def test_release_comparison_preserves_weight_and_sample_limits():
    text = (ROOT / "docs/BENCHMARKS.md").read_text()
    for required in (
        "135.2 / 200.4",
        "118.0 / 150.6",
        "75.5 / 116.7",
        "142.2 / 200.4",
        "84.1 / 150.6",
        "94.8 / 116.7",
        "three qualifying interleaved",
        "one prompt per depth",
        "Different weights",
        "zero cached tokens at 128K",
        "24.0 / 42.3",
        "screenshots remain pending",
    ):
        assert required in text
