"""APC must leave unsupported VLM request semantics on the existing path."""

from yunshu_engine.vlm_engine import VLMEngine


def _eligible(**changes):
    engine = object.__new__(VLMEngine)
    engine._apc_backend = object()
    params = dict(
        temperature=0,
        top_p=1,
        top_k=0,
        min_p=0,
        repetition_penalty=1,
        stop=None,
        stop_token_ids=None,
        enable_thinking=False,
        logprobs=False,
        top_logprobs=None,
        kwargs={"apc_allowed": True},
    )
    params.update(changes)
    return engine._apc_text_eligible(**params)


def test_plain_greedy_text_can_use_apc():
    assert _eligible()


def test_apc_does_not_steal_unsupported_request_modes():
    assert not _eligible(kwargs={"apc_allowed": False})  # tools/media gate
    assert not _eligible(
        kwargs={"apc_allowed": True, "json_schema": {"type": "object"}}
    )
    assert not _eligible(kwargs={"apc_allowed": True, "lora_adapter": object()})
    assert not _eligible(kwargs={"apc_allowed": True, "spec_decode": True})
    assert not _eligible(temperature=0.5)
    assert not _eligible(enable_thinking=True)
    assert not _eligible(stop=["END"])
    assert not _eligible(logprobs=True)
