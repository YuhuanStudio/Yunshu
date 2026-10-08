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
