import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "research"))
import decision_table as dt  # noqa: E402


def _arm(path, rows, tps, sha="a", *, complete=True, contended=False):
    recs = [
        {
            "part": "decode",
            "ctx": 1,
            "kind": "k",
            "phase": "warm",
            "dec_tps": tps,
            "sha": sha,
        }
    ]
    if complete:
        recs.append({"complete": True, "mode": "mtp", "contended": contended})
    path.write_text("\n".join(json.dumps(r) for r in recs))


def test_copy_table_flags_digest_changes_and_rejects_bad_arms(tmp_path):
    stem = str(tmp_path / "s")
    _arm(tmp_path / "s-w0-r0.jsonl", 0, 50.0)
    _arm(tmp_path / "s-w16-r0.jsonl", 16, 70.0, sha="b")
    _arm(tmp_path / "s-w8-r0.jsonl", 8, 60.0, complete=False)
    _arm(tmp_path / "s-w12-r0.jsonl", 12, 61.0, contended=True)
    lines, rejected = dt.copy_table([stem])
    assert "16: 70.0 (1) DIGEST DIFFERS" in lines[1]
    assert "8:" not in lines[1] and "12:" not in lines[1]
    assert len(rejected) == 2
