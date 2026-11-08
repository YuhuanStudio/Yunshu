"""The M3 correctness script's pure helpers (the server runs themselves are M3 jobs)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "pb2g_m3_check", ROOT / "scripts/research/pb2g_m3_check.py"
)
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)


def _run(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text(json.dumps({"rows": rows}))
    return str(p)


def test_compare_reports_each_differing_field(tmp_path):
    a = _run(
        tmp_path, "a.json", {"greedy": {"texts": ["x", "y"]}, "image": {"text": "red"}}
    )
    b = _run(
        tmp_path, "b.json", {"greedy": {"texts": ["x", "z"]}, "image": {"text": "red"}}
    )
    assert set(chk.compare(a, b)) == {"greedy.texts"}
    assert chk.compare(a, a) == {}


def test_text_of_and_image_helper():
    assert chk.text_of({"choices": [{"message": {"content": None}}]}) == ""
    assert chk.png_data_url().startswith("data:image/png;base64,")
