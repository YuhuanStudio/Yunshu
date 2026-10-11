import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "benchmarks_doc",
    Path(__file__).resolve().parents[2] / "scripts/dev/benchmarks_doc.py",
)
doc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doc)


def stat(values, measured=True):
    samples = [
        {"value": v, "version": "1", "mode": "dflash", "weights_vs_oQ4e": "same"}
        for v in values
    ]
    if measured:
        s = sorted(values)
        return {"status": "measured", "median": s[1], "mad": 1.0, "samples": samples}
    return {"status": "unknown", "samples": samples}


def board():
    item = {
        "ctx": 1024,
        "kind": "prose",
        "metric": "decode_cold_tps",
        "higher_is_better": True,
        "engines": {
            "yunshu-new": stat([80, 81, 82]),
            "splash": stat([90, 91, 92]),
            "omlx": stat([50], measured=False),
        },
    }
    return {"items": [item], "head_to_head": {}, "summary": "s", "verdict": "v"}


def test_best_is_bold_and_provisional_is_unranked():
    text = doc.render_table(
        board(), "decode_cold_tps", "T", "{:.3g}", list(doc.ENGINE_ORDER)
    )
    row = next(line for line in text.splitlines() if line.startswith("| 1K prose"))
    assert "**91 ±1 (n=3)**" in row and "81 ±1 (n=3)" in row
    assert "(50, n=1)" in row and "**(50" not in row
    assert row.count("unknown") == 4
    assert "Splash†" in text


def test_empty_metric_says_unknown_and_doc_has_no_private_paths():
    b = board()
    assert "all cells are unknown" in doc.render_table(
        b, "ttft_cold_s", "T", "{:.3g}", ["yunshu-new"]
    )
    out = doc.render(b, "# Intro")
    assert "/Volumes" not in out and "/Users" not in out
    assert "| Yunshu | 1 | dflash | same |" in out
