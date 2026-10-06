import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "longgap_agg", Path(__file__).parents[2] / "scripts" / "research" / "longgap_agg.py"
)
agg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agg)


def _file(path, ttfts, tps, done=True):
    recs = [{"part": "session"}]
    for ph, t in zip(("cold", "warm", "turn2"), ttfts, strict=True):
        recs.append(
            {
                "part": "decode",
                "ctx": 32768,
                "kind": "code",
                "phase": ph,
                "ttft_s": t,
                "dec_tps": tps,
                "sha": "a",
            }
        )
    if done:
        recs.append({"part": "part_done", "complete": True})
    path.write_text("\n".join(json.dumps(r) for r in recs))


def test_label_and_gap(tmp_path, capsys):
    _file(tmp_path / "tf-new-32768code-r0.jsonl", (40, 0.2, 3.0), 140)
    _file(tmp_path / "yunshu-32768code-r0.jsonl", (35, 0.2, 3.0), 70)
    agg.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert "GAP yunshu 32K code: decode 50%" in out
    assert "cold 114%" in out


def test_incomplete_file_is_reported_not_counted(tmp_path, capsys):
    _file(tmp_path / "yunshu-32768code-r0.jsonl", (1, 1, 1), 10, done=False)
    agg.main([str(tmp_path)])
    assert "INCOMPLETE" in capsys.readouterr().out
