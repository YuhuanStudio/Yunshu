"""Reasoning-parser gate (no model — pure parser).

Verifies the <think> reasoning parser correctly separates reasoning from the final
content (no reasoning leak into content), across: a normal think/answer, leading
content before the tag, multiple think cycles, and an unclosed tag. Guards the
reasoning-split protocol surface.

Run: PYTHONPATH=. uv run python scripts/verify_reasoning_parser.py
"""
from __future__ import annotations

import sys

from yunshu_engine.reasoning_parser import get_reasoning_parser


def _ra(text, model="Qwen3.5-0.8B"):
    o = get_reasoning_parser(model).parse(text)
    r = (getattr(o, "reasoning", None) or getattr(o, "reasoning_content", None) or "")
    c = (getattr(o, "content", None) or getattr(o, "text", None) or "")
    return (r or "").strip(), (c or "").strip()


def main() -> int:
    fails = 0
    cases = [
        ("normal", "<think>step one then two</think>The answer is 4.",
         lambda r, c: "step one" in r and "answer is 4" in c and "step one" not in c),
        ("leading content", "Sure.<think>hidden reasoning</think>Final reply.",
         lambda r, c: "hidden reasoning" in r and "hidden reasoning" not in c
                      and "Final reply" in c),
        ("multi-cycle", "<think>a</think>mid<think>b</think>end",
         lambda r, c: "a" in r and "b" in r and "mid" in c and "end" in c
                      and "a" not in c.replace("answer", "")),
        ("unclosed", "<think>open-ended reasoning with no close tag",
         lambda r, c: "open-ended reasoning" in r and c == ""),
    ]
    for label, text, check in cases:
        r, c = _ra(text)
        ok = check(r, c)
        print(f"  {'OK ' if ok else 'BAD'} {label}: reasoning={r[:40]!r} content={c[:40]!r}")
        if not ok:
            fails += 1
    print(f"RESULT: {len(cases) - fails} passed, {fails} failed")
    print("PASS" if fails == 0 else "FAIL")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
