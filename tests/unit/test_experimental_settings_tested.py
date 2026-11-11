"""Every experimental setting is referenced by at least one unit test (a flag no test flips ships unproven)."""

from pathlib import Path

from yunshu_engine import settings

_TESTS = Path(__file__).resolve().parent


def _unreferenced(names, text):
    return sorted(n for n in names if n not in text)


def test_every_experimental_setting_is_referenced_by_a_test():
    names = [
        s.name
        for s in settings.REGISTRY.values()
        if getattr(s, "stability", "") == "experimental"
    ]
    assert names, "no experimental settings found: the registry accessor changed"
    text = "\n".join(
        p.read_text(errors="ignore")
        for p in _TESTS.rglob("*.py")
        if p.name != Path(__file__).name
    )
    assert _unreferenced(names, text) == []


def test_unreferenced_helper_detects_a_missing_name():
    assert _unreferenced(["YUNSHU_A", "YUNSHU_B"], "x YUNSHU_A y") == ["YUNSHU_B"]
