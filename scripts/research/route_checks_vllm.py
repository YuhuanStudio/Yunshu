# Upstream (inspired): vllm-project/vllm (Apache-2.0) tests/entrypoints/openai/chat_completion/test_chat.py @ 68088ed
"""Request-validation and error-shape checks ported from vLLM's entrypoint tests (tests/entrypoints/
openai: test_chat.py, test_chat_completion.py, test_chat_logit_bias_validation.py,
test_non_object_body_validation.py, completion/test_completion.py, test_prompt_validation.py).
Each probe is a request a strict OpenAI server must refuse with a 4xx in the OpenAI error shape
(never a 500, a hang or a silent 200). Imported at the bottom of route_checks.py."""

from __future__ import annotations

from route_checks import Ctx, check, err_ok, expect

MSG = [{"role": "user", "content": "hi"}]


def _chat(**kw):
    return {"messages": MSG, "max_tokens": 4, **kw}


# (label, path, body-builder(model) -> kwargs for httpx) ; every one must answer 400..499.
def _bad_chat_probes():
    yield "stream_options_without_stream", _chat(stream_options={"include_usage": True})
    yield "stream_options_null_usage", _chat(stream_options={"include_usage": None})
    yield "top_logprobs_21", _chat(logprobs=True, top_logprobs=21)
    yield "top_logprobs_30_stream", _chat(logprobs=True, top_logprobs=30, stream=True)
    yield "top_logprobs_negative", _chat(logprobs=True, top_logprobs=-1)
    yield "seed_above_int64", _chat(seed=2**63)
    yield "seed_below_int64", _chat(seed=-(2**63) - 1)
    yield "logit_bias_non_int_key", _chat(logit_bias={"not_a_token_id": 5})
    yield "logit_bias_non_numeric_value", _chat(logit_bias={"1": "not_a_number"})
    yield "logit_bias_out_of_range_value", _chat(logit_bias={"1": 500})
    yield "structured_regex_invalid", _chat(structured_outputs={"regex": "[.*"})
    yield "structured_grammar_empty", _chat(structured_outputs={"grammar": ""})
    yield (
        "structured_json_invalid_schema",
        _chat(
            structured_outputs={"json": {"type": "object", "properties": {"a": "bar"}}}
        ),
    )
    yield "response_format_bad_type", _chat(response_format={"type": "nope"})
    yield "n_zero", _chat(n=0)
    yield "n_negative", _chat(n=-1)
    yield "max_tokens_negative", _chat(max_tokens=-1)
    yield "temperature_negative", _chat(temperature=-1)
    yield "top_p_above_one", _chat(top_p=2.0)
    yield "presence_penalty_range", _chat(presence_penalty=5)
    yield "messages_empty", {"messages": [], "max_tokens": 4}
    yield "messages_missing", {"max_tokens": 4}
    yield "messages_not_list", {"messages": "hi", "max_tokens": 4}
    yield "role_invalid", {"messages": [{"role": "robot", "content": "x"}]}
    yield "content_wrong_type", {"messages": [{"role": "user", "content": 7}]}
    yield "stop_wrong_type", _chat(stop=5)
    yield (
        "tool_choice_unknown_function",
        _chat(
            tools=[
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {"type": "object"}},
                }
            ],
            tool_choice={"type": "function", "function": {"name": "nope"}},
        ),
    )
    yield "tool_choice_required_no_tools", _chat(tool_choice="required")
    yield "tools_bad_type", _chat(tools=[{"type": "nope"}])
    yield "prompt_logprobs_negative", _chat(prompt_logprobs=-1)
    yield "json_schema_without_schema", _chat(response_format={"type": "json_schema"})


def _bad_completion_probes():
    yield "prompt_missing", {"max_tokens": 4}
    yield "prompt_null", {"prompt": None, "max_tokens": 4}
    yield "prompt_empty_list", {"prompt": [], "max_tokens": 4}
    yield "prompt_token_negative", {"prompt": [-5, 3], "max_tokens": 4}
    yield (
        "stream_options_without_stream",
        {
            "prompt": "hi",
            "max_tokens": 4,
            "stream_options": {"include_usage": True},
        },
    )
    yield "logprobs_negative", {"prompt": "hi", "max_tokens": 4, "logprobs": -1}
    yield "n_zero", {"prompt": "hi", "max_tokens": 4, "n": 0}
    yield "best_of_below_n", {"prompt": "hi", "max_tokens": 4, "n": 3, "best_of": 1}


def _non_object_bodies():
    yield "list", b'["not","an","object"]'
    yield "number", b"42"
    yield "null", b"null"
    yield "string", b'"hi"'
    yield "bad_json", b"this is not valid json{{{"
    yield "empty", b""


def _sweep(c: Ctx, path: str, probes, base: dict, family="openai", tolerate=()):
    bad = []
    for label, body in probes:
        if label in tolerate:
            continue
        payload = {"model": c.model, **body}
        try:
            r = c.req("POST", path, json=payload)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{label}: transport {type(e).__name__}")
            continue
        if not 400 <= r.status_code < 500:
            bad.append(f"{label}: {r.status_code} {r.text[:80]!r}")
            continue
        try:
            err_ok(r, family)
        except AssertionError as e:
            bad.append(f"{label}: {e}")
    return bad


@check("vllm_chat_validation", "POST /v1/chat/completions", served=False)
def _chat_validation(c: Ctx):
    bad = _sweep(c, "/v1/chat/completions", _bad_chat_probes(), {})
    c.notes["vllm_chat_validation"] = f"{len(bad)} gaps"
    expect(not bad, "accepted or mis-shaped: " + " | ".join(bad))


@check("vllm_completion_validation", "POST /v1/completions", served=False)
def _completion_validation(c: Ctx):
    bad = _sweep(c, "/v1/completions", _bad_completion_probes(), {})
    expect(not bad, "accepted or mis-shaped: " + " | ".join(bad))


@check(
    "vllm_non_object_body",
    "POST /v1/chat/completions",
    "POST /v1/completions",
    served=False,
)
def _non_object(c: Ctx):
    bad = []
    for path in ("/v1/chat/completions", "/v1/completions", "/v1/responses"):
        for label, raw in _non_object_bodies():
            r = c.req(
                "POST", path, content=raw, headers={"content-type": "application/json"}
            )
            if not 400 <= r.status_code < 500:
                bad.append(f"{path} {label}: {r.status_code}")
                continue
            try:
                err_ok(r, "openai")
            except AssertionError as e:
                bad.append(f"{path} {label}: {e}")
    expect(not bad, " | ".join(bad))


@check(
    "vllm_stream_usage",
    "POST /v1/chat/completions",
    "POST /v1/completions",
    served=True,
)
def _stream_usage(c: Ctx):
    """vLLM test_chat.py::test_stream_options: no usage without include_usage; with it every chunk
    has usage null and one final chunk has choices == [] and usage that adds up."""
    for kind in ("chat", "completion"):
        kw = {"messages": MSG} if kind == "chat" else {"prompt": "Say hi."}
        create = (
            c.oa.chat.completions.create if kind == "chat" else c.oa.completions.create
        )
        plain = list(create(model=c.model, max_tokens=6, stream=True, **kw))
        expect(
            all(ch.usage is None for ch in plain),
            f"{kind}: usage without include_usage",
        )
        empty = list(
            create(model=c.model, max_tokens=6, stream=True, stream_options={}, **kw)
        )
        expect(
            all(ch.usage is None for ch in empty),
            f"{kind}: usage with stream_options={{}}",
        )
        chunks = list(
            create(
                model=c.model,
                max_tokens=6,
                stream=True,
                stream_options={"include_usage": True},
                **kw,
            )
        )
        expect(len(chunks) >= 2, f"{kind}: {len(chunks)} chunks")
        expect(
            all(ch.usage is None for ch in chunks[:-1]),
            f"{kind}: usage on a non-final chunk",
        )
        last = chunks[-1]
        u = last.usage
        expect(u is not None, f"{kind}: no final usage chunk")
        expect(last.choices == [], f"{kind}: final usage chunk has choices")
        expect(
            u.prompt_tokens > 0
            and u.completion_tokens > 0
            and u.total_tokens == u.prompt_tokens + u.completion_tokens,
            f"{kind}: usage {u}",
        )
