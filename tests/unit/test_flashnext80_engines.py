import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import flashnext80_engines as fe  # noqa: E402


def test_parse_sse():
    assert fe.parse_sse_line(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n') == (
        "delta",
        "hi",
    )
    assert fe.parse_sse_line(
        b'data: {"choices":[],"usage":{"completion_tokens":5}}'
    ) == (
        "usage",
        {"completion_tokens": 5},
    )
    assert fe.parse_sse_line(b"data: [DONE]") == ("done", None)
    assert fe.parse_sse_line(b": keepalive") is None
    assert fe.parse_sse_line(b'data: {"choices":[{"delta":{}}]}') is None


def test_rate_and_cmd():
    assert (
        fe.decode_tok_s(
            {"decode_s": 2.0, "deltas": 0, "usage": {"completion_tokens": 21}}
        )
        == 10.0
    )
    assert fe.decode_tok_s({"decode_s": None, "deltas": 3, "usage": None}) is None
    assert "--no-drafts" in fe.server_cmd("tf-nodraft", "/m", 1, [])
    assert "--no-drafts" not in fe.server_cmd("tf-mtp", "/m", 1, ["--ple-on-ssd"])
    assert fe.arm_env("yunshu-mtp", [])["YUNSHU_VLM_DRAFT"] == "mtp"
    assert fe.arm_env("yunshu-off", [])["YUNSHU_VLM_DRAFT"] == "off"


def test_cli_parses_dash_values():
    import argparse  # noqa: F401

    ap_args = [
        "--arm",
        "tf-nodraft",
        "--model",
        "/m",
        "--out",
        "/o",
        "--tf-arg=--ple-on-ssd",
        "--env=A=B",
    ]
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        try:
            fe.main(ap_args + ["--help"])
        except SystemExit as e:
            assert e.code == 0


def test_gpu_command_lines_parse():
    a = fe.build_parser().parse_args(
        [
            "--arm",
            "tf-nodraft",
            "--model",
            "/m",
            "--ctx",
            "1024",
            "8192",
            "--reps",
            "1",
            "--tf-arg=--ple-on-ssd",
            "--env=TENSORFOLD_MEMORY_LIMIT_GB=107",
            "--out",
            "/o",
        ]
    )
    assert a.tf_arg == ["--ple-on-ssd"] and a.env == ["TENSORFOLD_MEMORY_LIMIT_GB=107"]
