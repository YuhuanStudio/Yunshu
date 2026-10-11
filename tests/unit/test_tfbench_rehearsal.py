import importlib.util
import pathlib

import pytest

spec = importlib.util.spec_from_file_location(
    "tfbench", pathlib.Path(__file__).parents[2] / "scripts/research/tfbench.py"
)
tfbench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tfbench)


def test_rehearsal_prefix_cuts_before_last_user():
    r = "<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\nB<|im_end|>\n<|im_start|>user\nC<|im_end|>\n"
    assert tfbench.rehearsal_prefix(r).endswith("B<|im_end|>\n")
    assert "C" not in tfbench.rehearsal_prefix(r)


def test_rehearsal_prefix_missing_marker_fails_closed():
    with pytest.raises(ValueError):
        tfbench.rehearsal_prefix("no user header here")


def test_followup_suite_passes_think_time_to_the_cells():
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts"))
    from verify import stages, suites

    class C:
        suite = suites.SUITES["followup"]

    args = stages._dec_args(C)
    assert args[args.index("--think-s") + 1] == "12.0"
    assert "--think-s" not in stages._dec_args(
        type("D", (), {"suite": suites.SUITES["long"]})
    )


METRICS = (
    'yunshu_next_turn_rehearsals_total{outcome="started"} 2\n'
    'yunshu_next_turn_rehearsals_total{outcome="completed"} 1\n'
)


def test_completed_rehearsals_parses_metrics():
    assert tfbench.completed_rehearsals(METRICS) == 1
    assert tfbench.completed_rehearsals("other 3\n") == 0


def _srv(env):
    return type("S", (), {"extra_env": env, "url": "http://x"})


def test_engagement_check_fails_closed_when_flag_on_and_zero_completed(monkeypatch):
    import io

    monkeypatch.setattr(
        tfbench.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(
            b'yunshu_next_turn_rehearsals_total{outcome="started"} 1\n'
        ),
    )
    a = type("A", (), {"think_s": 12.0})
    with pytest.raises(RuntimeError, match="never engaged"):
        tfbench.check_rehearsal_engaged(_srv({"YUNSHU_NEXT_TURN_PREFILL": "1"}), a)
    # flag off: no check; no think time: a mid-rehearsal cancel is legitimate
    assert tfbench.check_rehearsal_engaged(_srv({}), a) is None
    assert (
        tfbench.check_rehearsal_engaged(
            _srv({"YUNSHU_NEXT_TURN_PREFILL": "1"}), type("B", (), {"think_s": 0.0})
        )
        == 0
    )


def test_engagement_check_passes_when_completed(monkeypatch):
    import io

    monkeypatch.setattr(
        tfbench.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(METRICS.encode())
    )
    a = type("A", (), {"think_s": 12.0})
    assert (
        tfbench.check_rehearsal_engaged(_srv({"YUNSHU_NEXT_TURN_PREFILL": "1"}), a) == 1
    )


def test_boundary_reference_mode_is_not_checked(monkeypatch):
    a = type("A", (), {"think_s": 12.0})
    assert (
        tfbench.check_rehearsal_engaged(
            _srv({"YUNSHU_NEXT_TURN_PREFILL": "boundary"}), a
        )
        is None
    )


def test_arrival_offsets_are_uniform_with_a_control_after_the_window():
    offs = tfbench.arrival_offsets(3.0, 6)
    assert offs[:-1] == [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
    assert offs[-1] == 9.0


def test_arrive_part_is_registered_and_parses():
    a = tfbench.parse_args(
        [
            "--engine",
            "yunshu",
            "--part",
            "arrive",
            "--out",
            "x",
            "--arrive-mode",
            "adopt",
        ]
    )
    assert a.part == "arrive" and a.arrive_mode == "adopt" and a.arrive_n == 8


def test_arrive_requires_engagement_even_without_think_time(monkeypatch):
    import io

    monkeypatch.setattr(
        tfbench.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(
            b'yunshu_next_turn_rehearsals_total{outcome="started"} 1\n'
        ),
    )
    a = type("A", (), {"think_s": 0.0, "require_engaged": True})
    with pytest.raises(RuntimeError, match="never engaged"):
        tfbench.check_rehearsal_engaged(_srv({"YUNSHU_NEXT_TURN_PREFILL": "exact"}), a)


def test_part_arrive_runs_end_to_end_against_a_fake_server(monkeypatch, tmp_path):
    import io
    import json as _json

    sent = []

    def fake_send(url, body, *a, **k):
        sent.append(body)
        n = body.get("max_tokens", 0)
        return {
            "ttft_s": 0.1,
            "_text": "word " * n,
            "cached": 1,
            "pt": 2,
            "ct": n,
            "finish": "length",
        }

    monkeypatch.setattr(tfbench, "send", fake_send)
    monkeypatch.setattr(tfbench, "load_prompt", lambda name: "prompt")
    monkeypatch.setattr(tfbench, "check_decode_len", lambda *a, **k: None)
    monkeypatch.setattr(tfbench.time, "sleep", lambda s: None)
    metrics = (
        'yunshu_next_turn_rehearsals_total{outcome="completed"} 3\n'
        'yunshu_next_turn_rehearsals_total{outcome="preempted"} 2\n'
    )
    monkeypatch.setattr(
        tfbench.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(metrics.encode())
    )
    s = type(
        "S",
        (),
        {
            "url": "http://x",
            "model": "m",
            "extra_env": {"YUNSHU_NEXT_TURN_PREFILL": "exact"},
        },
    )
    for mode in ("diverge", "adopt"):
        a = tfbench.parse_args(
            [
                "--engine",
                "yunshu",
                "--part",
                "arrive",
                "--out",
                "x",
                "--arrive-mode",
                mode,
                "--arrive-n",
                "2",
                "--decode-tokens",
                "5",
            ]
        )
        out = tmp_path / f"{mode}.jsonl"
        with open(out, "w") as fh:
            tfbench.part_arrive(s, fh, a)
        rows = [_json.loads(line) for line in out.read_text().splitlines()]
        summary = rows[-1]
        assert summary["role"] == "summary" and summary["preempted"] == 2
        assert summary["rehearsals_completed"] == 3
        assert [r["role"] for r in rows].count("trial") == 3  # 2 offsets + the control
