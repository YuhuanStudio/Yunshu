"""VLM streaming dropped pre-stop content when a stop string completed inside a
content-bearing token.

StopHoldbackBuffer.feed() RELEASES the text before the stop (and removes it from the
buffer); take_stopped() then returns only what remains up to the stop. The VLM streaming
paths did `_hb.feed(seg); token_text = _hb.take_stopped()` — discarding feed()'s return. So
for a segment like "goodbyeEND" with stop "END", "goodbye" was permanently lost from the
append-only SSE stream. The text engine emits BOTH; the VLM (and the dead MTP) paths now do
too.
"""

from __future__ import annotations

from yunshu_engine.text_utils import StopHoldbackBuffer


def test_holdback_loses_content_when_only_take_stopped_used():
    # demonstrates the bug the fix addresses
    hb = StopHoldbackBuffer(["END"])
    emit = hb.feed("goodbyeEND")
    tail = hb.take_stopped()
    assert emit == "goodbye" and tail == ""
    # the OLD code emitted only `tail` → "goodbye" lost; the NEW code emits emit+tail
    assert emit + tail == "goodbye"
