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
            "All three locales, light/dark and mobile images were visually reviewed.",
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
        "All 78 console screenshots now show the qualified",
    ):
        assert required in text


def test_published_capture_receipt_matches_every_real_image():
    import hashlib
    import json
    import statistics

    from scripts.research.docs016_console import validate_long_session

    receipt = json.loads(
        (ROOT / "docs/reports/CONSOLE_CAPTURE_2026-10-19.json").read_text()
    )
    assert receipt["complete"] and receipt["pass"]
    assert receipt["gpuq"]["state"] == "done" and receipt["gpuq"]["rc"] == 0
    assert receipt["gpuq"]["contended"] is False
    assert receipt["spec_mode"] == "mtp" and receipt["failures"] == 0
    rows = receipt["long_session_usage"]
    assert validate_long_session(rows) == []
    assert receipt["median_reply_tokens"] == statistics.median(
        row["completion_tokens"] for row in rows
    )
    assert receipt["visual_review_images"] == len(receipt["images"]) == 78
    expected = {
        str(p.relative_to(ROOT))
        for p in (ROOT / "docs/images/console").glob("*/*.webp")
    }
    assert {image["path"] for image in receipt["images"]} == expected
    for image in receipt["images"]:
        raw = (ROOT / image["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == image["sha256"]
        assert len(raw) == image["size"] <= 300_000
