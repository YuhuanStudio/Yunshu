"""Unit tests for StopHoldbackBuffer — multi-token stop-string hold-back.

Pure logic, no model. Pins the streaming-correctness rule: a multi-token stop
string must never leak its prefix to the client (SSE is append-only).
"""

from __future__ import annotations

from yunshu_engine.text_utils import StopHoldbackBuffer


def _drive(stops, tokens):
    """Feed token texts; return (emitted_before_stop, stopped_tail)."""
    buf = StopHoldbackBuffer(stops)
    emitted = []
    for t in tokens:
        emitted.append(buf.feed(t))
    return "".join(emitted), buf


def test_no_stops_is_passthrough():
    buf = StopHoldbackBuffer(None)
    assert buf.feed("hello") == "hello"
    assert buf.feed(" world") == " world"
    assert buf.flush() == ""


def test_multi_token_stop_prefix_not_leaked():
    # stop="\n\n" arriving as two "\n" tokens: the first "\n" must be held back.
    emitted, buf = _drive(["\n\n"], ["a", "\n", "\n"])
    assert emitted == "a"  # NOT "a\n"
    # stop completes → take_stopped trims "\n\n", leaving nothing extra
    assert buf.take_stopped() == ""


def test_single_token_stop_with_leading_text():
    # token "doneSTOP" with stop "STOP" → "done" is safe now, "STOP" is held
    # (a full stop-prefix); take_stopped then trims it to "".
    buf = StopHoldbackBuffer(["STOP"])
    assert buf.feed("doneSTOP") == "done"
    assert buf.take_stopped() == ""


def test_text_after_potential_prefix_flushes():
    # "ab\nc": the "\n" looks like a stop start, but "c" proves it isn't.
    buf = StopHoldbackBuffer(["\n\n"])
    out1 = buf.feed("ab\n")  # hold back "\n"
    assert out1 == "ab"
    out2 = buf.feed("c")  # "\nc" can't be a stop prefix → flush both
    assert out2 == "\nc"


def test_partial_then_complete_across_three_tokens():
    # stop "###" across "#","#","#".
    buf = StopHoldbackBuffer(["###"])
    assert buf.feed("x#") == "x"  # hold "#"
    assert buf.feed("#") == ""  # "##" still a prefix
    assert buf.feed("#") == ""  # "###" now full match (held)
    assert buf.take_stopped() == ""


def test_flush_emits_held_back_when_no_stop():
    # ends with a stop-prefix that never completes → flush emits it.
    buf = StopHoldbackBuffer(["END"])
    assert buf.feed("textEN") == "text"  # hold "EN"
    assert buf.flush() == "EN"


def test_longest_prefix_among_multiple_stops():
    buf = StopHoldbackBuffer(["ab", "abcd"])
    # "xabc": suffix "abc" is a prefix of "abcd" (len 3) → hold "abc".
    assert buf.feed("xabc") == "x"
    assert buf.feed("d") == ""  # "abcd" completes
    assert buf.take_stopped() == ""


def test_unrelated_text_passes_through_immediately():
    buf = StopHoldbackBuffer(["STOP"])
    assert buf.feed("hello world") == "hello world"
    assert buf.feed(" more") == " more"


def test_take_stopped_without_match_returns_buffer():
    buf = StopHoldbackBuffer(["STOP"])
    buf.feed("ST")  # held back (prefix of STOP)
    # stop fired by a stop_token_id, not a string match → no suffix to trim
    assert buf.take_stopped() == "ST"


# ---------------------------------------------------------------------------
# feed_lp(): logprob-aware hold-back. Pins that, with logprobs on, a multi-token
# stop prefix STILL never leaks AND each token's logprob entry is emitted
# exactly once aligned to the chunk carrying its text.
# ---------------------------------------------------------------------------


def _drive_lp(stops, tokens):
    """Feed (text, lp) tokens through feed_lp; return list of (text, lp) chunks."""
    buf = StopHoldbackBuffer(stops)
    out = []
    for i, t in enumerate(tokens):
        out.extend(buf.feed_lp(t, {"id": i}))
    return out, buf


def test_feed_lp_no_stops_is_passthrough():
    buf = StopHoldbackBuffer(None)
    assert buf.feed_lp("hello", {"id": 0}) == [("hello", {"id": 0})]
    assert buf.feed_lp("", {"id": 1}) == []  # empty text emits nothing


def test_feed_lp_no_leak_and_lp_aligned():
    # stop "\n\n" as two "\n" tokens, logprobs on. The first "\n" must be held
    # back (no leak), and each token's lp must appear exactly once.
    out, buf = _drive_lp(["\n\n"], ["a", "\n", "\n"])
    # token 0 "a" emits immediately with its lp; token 1 "\n" is held (stop
    # prefix) so nothing emits; token 2 completes the stop → still held.
    assert out == [("a", {"id": 0})]
    # the held "\n\n" is the stop → discarded, no leak.
    assert buf.take_stopped() == ""


def test_feed_lp_each_token_lp_emitted_once():
    # No stop fires; every token's text and lp must come out, in order, once.
    out, buf = _drive_lp(["STOP"], ["hello", " world", "!"])
    assert out == [("hello", {"id": 0}), (" world", {"id": 1}), ("!", {"id": 2})]
    assert buf.flush() == ""


def test_feed_lp_partial_token_split_lp_not_duplicated():
    # token 0 = "ab\n" with stop "\n\n": "ab" releases now (carries token 0's
    # lp), "\n" is held. token 1 = "c" proves no stop → "\nc" releases. The
    # already-emitted token 0 must NOT re-emit its lp; token 1 carries its own.
    buf = StopHoldbackBuffer(["\n\n"])
    out0 = buf.feed_lp("ab\n", {"id": 0})
    assert out0 == [("ab", {"id": 0})]
    out1 = buf.feed_lp("c", {"id": 1})
    # "\n" is the held tail of token 0 (lp already emitted → None);
    # "c" is token 1 (its lp).
    assert out1 == [("\n", None), ("c", {"id": 1})]


def test_feed_lp_held_token_lp_released_when_text_escapes():
    # token 0 = "EN" held (prefix of "END"); token 1 = "X" proves no stop.
    # token 0's lp was pending → it must ride the chunk that first reveals "E".
    buf = StopHoldbackBuffer(["END"])
    assert buf.feed_lp("EN", {"id": 0}) == []  # fully held
    out1 = buf.feed_lp("X", {"id": 1})
    assert out1 == [("EN", {"id": 0}), ("X", {"id": 1})]


def test_feed_lp_concatenation_matches_visible_text():
    # Invariant: concatenating emitted text (minus the discarded stop) equals
    # the user-visible output, and lp entries appear once each in order.
    out, buf = _drive_lp(["##"], ["foo", "#", "bar", "#", "#"])
    emitted_text = "".join(t for t, _ in out)
    lp_ids = [lp["id"] for _, lp in out if lp is not None]
    # "foo" out; "#" held; "bar" → "#bar" releases (# wasn't a stop, bar after);
    # then "#","#" form "##" → held → stop.
    assert emitted_text == "foo#bar"
    assert lp_ids == sorted(set(lp_ids))  # each id at most once, ascending
    assert buf.take_stopped() == ""


def test_stop_mid_segment_does_not_leak():
    """A complete stop landing MID-segment (one token decodes to 'aSTOPb') must NOT
    leak — the suffix-only holdback used to emit the whole thing. Now feed() emits
    only the pre-stop text, contains_stop() is True, take_stopped() find-truncates."""
    from yunshu_engine.text_utils import StopHoldbackBuffer

    b = StopHoldbackBuffer(["STOP"])
    out = b.feed("aSTOPb")
    assert out == "a", out  # only the pre-stop text is emitted
    assert "STOP" not in out
    assert b.contains_stop() is True  # the held "STOPb" contains a complete stop
    assert b.take_stopped() == ""  # find-truncate at the stop → nothing before it
    # 'done' before a mid-segment stop in ONE feed → emit 'done', stop dropped
    b2 = StopHoldbackBuffer(["STOP"])
    assert b2.feed("doneSTOPextra") == "done"
    assert b2.take_stopped() == ""


def test_stop_mid_segment_flush_drops_it():
    """If the engine never fires (held stop reaches flush), flush drops it — no leak."""
    from yunshu_engine.text_utils import StopHoldbackBuffer

    b = StopHoldbackBuffer(["STOP"])
    assert b.feed("aSTOPb") == "a"
    assert b.flush() == ""  # held 'STOPb' → flush find-truncates → drops it


def test_normal_holdback_unaffected():
    """The split-token suffix case and no-stop passthrough still work."""
    from yunshu_engine.text_utils import StopHoldbackBuffer

    b = StopHoldbackBuffer(["STOP"])
    assert b.feed("done") == "done"
    assert b.feed("ST") == ""
    assert b.feed("OP") == ""
    assert b.take_stopped() == ""
    assert StopHoldbackBuffer(["STOP"]).feed("hello world") == "hello world"
