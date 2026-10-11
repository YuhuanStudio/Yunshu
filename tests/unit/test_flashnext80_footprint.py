import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import flashnext80_footprint as ff  # noqa: E402


def req(tokens=3, rounds=0, engaged=True):
    return {
        "kind": "request",
        "tokens": tokens,
        "x_yunshu": {"speculative": {"rounds": rounds}},
        "ple": {"engaged": engaged},
        "peak_delta_gb": 70.0,
        "steady_delta_gb": 69.0,
    }


def test_verdict_fail_closed():
    assert ff.verdict([], False, False) == ["no request rows"]
    assert ff.verdict([req()], True, False) == []
    assert "PLE-on-SSD counters did not move" in ff.verdict(
        [req(engaged=False)], True, False
    )
    assert "MTP engaged zero verify rounds" in ff.verdict([req()], False, True)
    assert ff.verdict([req(rounds=2)], False, True) == []
    assert "a request generated no tokens" in ff.verdict([req(tokens=0)], False, False)


def test_summary():
    s = ff.summarize([req(), req()], 10e9)
    assert s["baseline_gb"] == 10.0 and s["peak_delta_gb"] == 70.0


def test_gpu_command_lines_parse():
    """Every command line the submit helpers build must parse (three GPU jobs died on argparse once)."""
    a = ff.build_parser().parse_args(
        [
            "--tree",
            "/t",
            "--model",
            "/m",
            "--sizes",
            "1024",
            "8192",
            "--max-tokens",
            "160",
            "--env",
            "YUNSHU_VLM_DRAFT=mtp",
            "--env",
            "YUNSHU_MTP_BLOCK_SIZE=4",
            "--require-ple",
            "--require-spec",
            "--out",
            "/o",
        ]
    )
    assert (
        a.env == ["YUNSHU_VLM_DRAFT=mtp", "YUNSHU_MTP_BLOCK_SIZE=4"] and a.require_spec
    )
