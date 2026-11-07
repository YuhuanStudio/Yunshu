"""The patched-symbol fingerprint must not depend on the interpreter or on formatting, only on code."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_upstream", ROOT / "scripts/vendor/check_upstream.py"
)
cu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cu)

SNIP = '''class K:
    def f(self, x, *, y=2):  # a comment
        """doc"""
        s = f"a{x!r:>{y}}b"
        return [x + 1,
                s]
'''
# Recorded under Python 3.13 and re-checked under 3.12 (uv): identical.
SNIP_DIGEST = "f5fd9eff53dbcd27b0d0dcb2338d3aa9ddf44f824cfb9edd99064389402b479b"


def test_fixed_snippet_digest_is_stable():
    assert cu.source_fingerprint(SNIP, "K.f")[0] == SNIP_DIGEST


def test_comments_layout_and_blank_lines_do_not_change_it():
    variant = SNIP.replace("# a comment", "# other").replace(
        "return [x + 1,\n                s]", "return [x + 1, s]\n"
    )
    assert cu.source_fingerprint(variant, "K.f")[0] == SNIP_DIGEST


def test_a_real_code_change_changes_it():
    assert (
        cu.source_fingerprint(SNIP.replace("x + 1", "x + 2"), "K.f")[0] != SNIP_DIGEST
    )


def test_missing_symbol_is_none():
    assert cu.source_fingerprint(SNIP, "K.nope") is None
