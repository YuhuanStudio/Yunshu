import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import flashnext80_gdn_identity as gi  # noqa: E402


def test_parser():
    a = gi.build_parser().parse_args(
        ["--model", "/m", "--rows", "2", "3", "--out", "/o"]
    )
    assert a.rows == [2, 3] and a.layers == 3
