"""Published benchmark copies preserve qualified evidence and failed coverage."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUMBER = re.compile(r"\d+(?:\.\d+)?")
PAGES = ("", ".zh-TW", ".zh-CN")


def table_rows(text):
    return [line for line in text.splitlines() if line.startswith("| ")]


def numeric_rows(text):
    return [
        tuple(NUMBER.findall(line)) for line in table_rows(text) if NUMBER.search(line)
    ]


def page_body(text):
    return text.split("\n---\n", 1)[1] if text.startswith("---\n") else text


def test_console_benchmark_pages_match_benchmarks_md():
    source = (ROOT / "docs/BENCHMARKS.md").read_text()
    source_numbers = set(NUMBER.findall(source))
    source_rows = numeric_rows(source)
    for locale in PAGES:
        page = (ROOT / f"frontend/docs/guides/benchmarks{locale}.mdx").read_text()
        body = page_body(page)
        assert set(NUMBER.findall(body)) == source_numbers, locale
        page_rows = numeric_rows(body)
        assert all(row in page_rows for row in source_rows), locale
        for required in (
            "2026-11-08",
            "gaps_full_1109",
            "0.983",
            "-92.9%",
            "74.3",
            "-54.6%",
        ):
            assert required in body, (locale, required)


def test_release_015_history_rows_unchanged():
    history = (ROOT / "docs/reports/RELEASE_015_BENCHMARKS.md").read_text()
    assert len(table_rows(history)) == 68
    assert "no part_done (incomplete job)" in history
    assert "unusable: garbage output" in history
    assert "overall verdict was FAIL" in history


def test_release_comparison_preserves_weight_and_sample_limits():
    text = (ROOT / "docs/BENCHMARKS.md").read_text()
    for required in (
        "2026-11-04 to 2026-11-08",
        "Yunshu 0.1.7.dev0",
        "191.2 vs 238.7",
        "107.5 vs 122.7",
        "80.9 vs 96.5",
        "32.846 vs 30.570",
        "114 comparable rows",
        "Kapitsa 0.2.26",
        "0 cached tokens",
        "0.983 s",
        "Flash-Next (Qwen4) is not part of this comparison",
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
