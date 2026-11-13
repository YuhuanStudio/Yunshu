"""Decode-tail store policy: what each API shape says about the client echoing reasoning, the skip rule,
and the runner gate."""

import logging
from types import SimpleNamespace

from yunshu_engine import decode_tail_policy as dtp
from yunshu_engine.vlm_batch_runner import VLMBatchRunner

OPEN, CLOSE = 248068, 248069


class _Tok:
    unk_token_id = 0

    def convert_tokens_to_ids(self, tok):
        return {"<think>": OPEN, "</think>": CLOSE}[tok]


def _gate(runner, echo, apc=None, prompt=(5, OPEN, 9), generated=(7, 8, CLOSE, 11)):
    row = SimpleNamespace(
        req=SimpleNamespace(
            ids=list(prompt), handle=SimpleNamespace(client_echoes_reasoning=echo)
        ),
        context=list(prompt) + list(generated),
        apc=apc,
    )
    return VLMBatchRunner._reasoning_tail_skipped(runner, row)


# chat completions ---------------------------------------------------------


def test_chat_history_without_reasoning_is_not_echoed():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert dtp.chat_echoes_reasoning(msgs) is False


def test_chat_history_with_reasoning_content_is_echoed():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
    ]
    assert dtp.chat_echoes_reasoning(msgs) is True


def test_chat_history_with_reasoning_alias_is_echoed():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning": "r"},
    ]
    assert dtp.chat_echoes_reasoning(msgs) is True


def test_chat_turn_one_has_no_history():
    assert dtp.chat_echoes_reasoning([{"role": "user", "content": "q"}]) is None


# responses ----------------------------------------------------------------


def test_responses_reasoning_item_in_history_is_echoed():
    from yunshu_gateway.routers.responses import ResponsesRequest, _convert_to_messages

    req = ResponsesRequest(
        model="m",
        input=[
            {"type": "message", "role": "user", "content": "q1"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "r1"}]},
            {"type": "message", "role": "assistant", "content": "a1"},
            {"type": "message", "role": "user", "content": "q2"},
        ],
    )
    assert dtp.chat_echoes_reasoning(_convert_to_messages(req)) is True


def test_responses_history_without_reasoning_is_not_echoed():
    from yunshu_gateway.routers.responses import ResponsesRequest, _convert_to_messages

    req = ResponsesRequest(
        model="m",
        input=[
            {"type": "message", "role": "user", "content": "q1"},
            {"type": "message", "role": "assistant", "content": "a1"},
            {"type": "message", "role": "user", "content": "q2"},
        ],
    )
    assert dtp.chat_echoes_reasoning(_convert_to_messages(req)) is False


# anthropic messages -------------------------------------------------------


def test_anthropic_thinking_block_is_echoed():
    msgs = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "r1", "signature": "s"},
                {"type": "text", "text": "a1"},
            ],
        },
        {"role": "user", "content": "q2"},
    ]
    assert dtp.anthropic_echoes_reasoning(msgs) is True


def test_anthropic_text_only_history_is_not_echoed():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "user", "content": "q2"},
    ]
    assert dtp.anthropic_echoes_reasoning(msgs) is False


def test_anthropic_turn_one_has_no_history():
    assert dtp.anthropic_echoes_reasoning([{"role": "user", "content": "q"}]) is None


# the skip rule -------------------------------------------------------------


def test_skip_only_for_unechoed_history_with_stranded_reasoning():
    prompt = [5, OPEN, 9]  # thinking on: the prompt left <think> open
    stranded = [7, 8, CLOSE, 11]
    assert dtp.skip_decode_tail(False, prompt, stranded, OPEN, CLOSE)
    assert not dtp.skip_decode_tail(
        True, prompt, stranded, OPEN, CLOSE
    )  # client echoes: store
    assert not dtp.skip_decode_tail(
        None, prompt, stranded, OPEN, CLOSE
    )  # turn 1: store


def test_no_skip_when_the_reply_has_no_reasoning():
    assert not dtp.skip_decode_tail(False, [5, OPEN, 9], [CLOSE, 11], OPEN, CLOSE)


def test_no_skip_for_a_thinking_off_prompt():
    assert not dtp.skip_decode_tail(
        False, [5, OPEN, CLOSE, 9], [7, 8, CLOSE, 11], OPEN, CLOSE
    )


def test_no_skip_without_think_ids():
    assert not dtp.skip_decode_tail(False, [5, OPEN, 9], [7, CLOSE], None, CLOSE)


def test_reasoning_token_ids_reject_unknown_tokens():
    class Tok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, tok):
            return {"<think>": OPEN, "</think>": 0}[tok]

    assert dtp.reasoning_token_ids(Tok()) == (OPEN, None)


# the runner gate -----------------------------------------------------------


def test_runner_gate_skips_unechoed_reasoning_and_counts_it(caplog):
    rejects = []
    stats = SimpleNamespace(record_reject=lambda reason, **kw: rejects.append(reason))
    apc = SimpleNamespace(manager=SimpleNamespace(stats=stats))
    runner = SimpleNamespace(processor=SimpleNamespace(tokenizer=_Tok()))
    dtp._LOGGED.discard("reasoning_unechoed")
    with caplog.at_level(logging.INFO, logger=dtp.logger.name):
        assert _gate(runner, False, apc=apc)
        assert not _gate(runner, True, apc=apc)
        assert not _gate(runner, None, apc=apc)
    assert rejects == ["decode_tail_reasoning"]
    assert (
        sum(
            "sends the history back without reasoning" in r.message
            for r in caplog.records
        )
        == 1
    )


def test_runner_gate_without_a_processor_keeps_the_store():
    runner = VLMBatchRunner.__new__(VLMBatchRunner)
    assert not _gate(runner, False)


# the tracker copies the reading onto the job -------------------------------


def test_set_request_echo_writes_the_tracker_info(monkeypatch):
    from yunshu_engine import request_tracker

    info = SimpleNamespace()
    token = request_tracker.current_request_info.set(info)
    try:
        dtp.set_request_echo(False)
    finally:
        request_tracker.current_request_info.reset(token)
    assert info.client_echoes_reasoning is False


def test_set_request_echo_reaches_a_generation_registered_earlier():
    from yunshu_engine import request_tracker

    cancel = SimpleNamespace()
    info = SimpleNamespace(gen=SimpleNamespace(cancel_event=cancel))
    token = request_tracker.current_request_info.set(info)
    try:
        dtp.set_request_echo(False)
    finally:
        request_tracker.current_request_info.reset(token)
    assert cancel.client_echoes_reasoning is False


def test_runner_gate_reads_the_reading_that_arrived_after_the_job_was_built():
    # the job is built before the router's reading lands on the cancel event it shares with the job
    runner = SimpleNamespace(processor=SimpleNamespace(tokenizer=_Tok()))
    job = SimpleNamespace(
        client_echoes_reasoning=None,
        cancel_event=SimpleNamespace(client_echoes_reasoning=False),
    )
    row = SimpleNamespace(
        req=SimpleNamespace(ids=[5, OPEN, 9], handle=job),
        context=[5, OPEN, 9, 7, 8, CLOSE, 11],
        apc=None,
    )
    assert VLMBatchRunner._reasoning_tail_skipped(runner, row)


def test_orphan_adoption_forwards_the_echo_reading_to_the_engine_cancel_event():
    # the engine's cancel event is a fresh one: only the attributes listed here cross to it
    from yunshu_engine import orphan_adoption

    assert "client_echoes_reasoning" in orphan_adoption.FORWARDED_CANCEL_ATTRS
