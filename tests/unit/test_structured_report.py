"""structured_report: what the request says about the constraint it actually got (direct unit test)."""

from yunshu_engine import request_tracker
from yunshu_engine.structured_report import backend_name, report


class _Info:
    structured_output = None


def test_not_requested_reports_no_constraint():
    value = report("mlx-lm", None)
    assert value == {
        "requested": False,
        "enforced": False,
        "engine": "mlx-lm",
        "grammar_backend": None,
        "reason": "not_requested",
    }


def test_requested_but_not_enforced_is_setup_failed():
    value = report("mlx-lm", {"type": "object"}, backend="LLGMatcher")
    assert value["requested"] is True
    assert value["enforced"] is False
    assert value["reason"] == "setup_failed"
    assert value["grammar_backend"] == "LLGMatcher"


def test_enforced_constraint_reports_enforced():
    value = report("vlm", {"type": "object"}, "Backend", enforced=True)
    assert value["enforced"] is True
    assert value["reason"] == "enforced"


def test_report_is_attached_to_the_current_request():
    info = _Info()
    token = request_tracker.current_request_info.set(info)
    try:
        value = report("mlx-lm", {"type": "object"}, enforced=True)
    finally:
        request_tracker.current_request_info.reset(token)
    assert info.structured_output is value


def test_backend_name_is_the_type_name_or_none():
    assert backend_name(None) is None
    assert backend_name(_Info()) == "_Info"
