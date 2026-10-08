from yunshu_engine import next_turn as nt

IM = 9


def _decode(ids):
    vocab = {
        9: "<|im_start|>",
        1: "user",
        2: "assistant",
        3: "\n",
        4: "A",
        5: "B",
        6: nt.SENTINEL,
    }
    return "".join(vocab[i] for i in ids)


def test_cut_before_sentinel_user_turn():
    ids = [9, 1, 3, 4, 9, 2, 3, 5, 9, 1, 3, 6, 9, 2, 3]
    assert nt.cut_before_user_turn(ids, IM, _decode) == ids[:8]


def test_cut_fails_closed():
    assert nt.cut_before_user_turn([9, 1, 3, 4], IM, _decode) is None
    assert nt.cut_before_user_turn([1, 2], None, _decode) is None
    # second-to-last header is not the sentinel user turn
    assert (
        nt.cut_before_user_turn([9, 1, 3, 4, 9, 2, 3, 5, 9, 2, 3], IM, _decode) is None
    )


def test_reply_text_drops_reasoning_and_strips():
    assert nt.reply_text([("think", "reasoning"), ("\n\nhi ", "normal")]) == "hi"


def test_eligibility():
    ok = dict(
        finish_reason="stop",
        completion_tokens=100,
        reply="x",
        has_media=False,
        has_tools=False,
        constrained=False,
    )
    assert nt.eligible(**ok)
    for k, v in (
        ("finish_reason", "error"),
        ("completion_tokens", 3),
        ("completion_tokens", 10**6),
        ("reply", ""),
        ("has_media", True),
        ("has_tools", True),
        ("constrained", True),
    ):
        assert not nt.eligible(**{**ok, k: v})


def test_with_reply_appends_assistant_and_sentinel():
    m = nt.with_reply([{"role": "user", "content": "q"}], "a")
    assert (
        m[-2] == {"role": "assistant", "content": "a"}
        and m[-1]["content"] == nt.SENTINEL
    )


def test_length_capped_reply_is_rehearsed():
    # the yv followup cells cap the first reply at max_tokens; it is re-sent unchanged
    assert nt.eligible(
        finish_reason="length",
        completion_tokens=2048,
        reply="x",
        has_media=False,
        has_tools=False,
        constrained=False,
    )


def test_rehearsal_length_ends_one_guard_after_the_stride_multiple():
    assert nt.rehearsal_length(34860, 2048, 1) == 34816 + 1
    assert (
        nt.rehearsal_length(34816, 2048, 1) == 34816 - 2048 + 1
    )  # guard would pass the prefix
    assert nt.rehearsal_length(1500, 2048, 1) is None
    assert nt.rehearsal_length(5000, 0, 1) == 5000  # no canonical grid: rehearse it all
