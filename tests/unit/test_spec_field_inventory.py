"""Every request parameter of the installed official SDKs is either modelled or listed here with a reason.

A new SDK release that adds a parameter fails this test, so the field is triaged (implement it, or write down why
it is ignored) instead of being silently dropped by pydantic's default `extra="ignore"`.
"""

import typing

import pytest

pytest.importorskip("openai")
pytest.importorskip("anthropic")

from anthropic.types.beta.message_create_params import (
    MessageCreateParamsBase as BetaMessages,
)  # noqa: E402
from anthropic.types.message_create_params import (
    MessageCreateParamsBase as Messages,  # noqa: E402
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase as Chat,
)  # noqa: E402
from openai.types.completion_create_params import (
    CompletionCreateParamsBase as Completions,
)  # noqa: E402
from openai.types.responses.response_create_params import (
    ResponseCreateParamsBase as Responses,
)  # noqa: E402

from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest  # noqa: E402
from yunshu_gateway.routers.chat import ChatCompletionRequest  # noqa: E402
from yunshu_gateway.routers.completions import CompletionRequest  # noqa: E402
from yunshu_gateway.routers.responses import ResponsesRequest  # noqa: E402

HOSTED = "hosted-platform concept with no local meaning (accepted, ignored)"
ACCEPTED_IGNORED = {
    "chat": {
        "audio": "audio output modality (use /v1/audio/speech or Realtime)",
        "modalities": "text only",
        "prediction": "predicted outputs: no local speed-up path",
        "store": HOSTED,
        "metadata": HOSTED,
        "service_tier": HOSTED,
        "moderation": HOSTED,
        "safety_identifier": HOSTED,
        "verbosity": "prompt-level hint for hosted models only",
        "web_search_options": "chat search models are hosted; web search is a Responses / Messages server tool here",
        "functions": "deprecated by OpenAI in favour of `tools`; KNOWN GAP (see AGENT_COMPAT.md)",
        "function_call": "deprecated by OpenAI in favour of `tool_choice`; KNOWN GAP",
    },
    "responses": {
        "access_programs": HOSTED,
        "moderation": HOSTED,
        "prompt": "stored prompt templates live on the hosted platform; KNOWN GAP",
    },
    "messages": {
        "diagnostics": HOSTED,
        "inference_geo": HOSTED,
        "user_profile_id": HOSTED,
        "workspace_id": HOSTED,
    },
    "messages-beta": {
        "diagnostics": HOSTED,
        "inference_geo": HOSTED,
        "user_profile_id": HOSTED,
        "workspace_id": HOSTED,
        "betas": "sent as the anthropic-beta header, never in the body",
        "compaction": "server-side compaction beta; Claude Code compacts client-side",
        "fallback_credit_token": HOSTED,
        "fallbacks": "model fallback list of the hosted API",
        "speed": "fast-mode tier of the hosted API",
    },
    "completions": {},
}
SURFACES = {
    "chat": (Chat, ChatCompletionRequest),
    "responses": (Responses, ResponsesRequest),
    "messages": (Messages, AnthropicMessagesRequest),
    "messages-beta": (BetaMessages, AnthropicMessagesRequest),
    "completions": (Completions, CompletionRequest),
}


@pytest.mark.parametrize("name", sorted(SURFACES))
def test_every_sdk_parameter_is_triaged(name):
    sdk, model = SURFACES[name]
    params = set(typing.get_type_hints(sdk))
    untriaged = params - set(model.model_fields) - set(ACCEPTED_IGNORED[name])
    assert not untriaged, (
        f"{name}: new SDK parameters neither modelled nor triaged: {sorted(untriaged)}"
    )
    stale = set(ACCEPTED_IGNORED[name]) & set(model.model_fields)
    assert not stale, (
        f"{name}: listed as ignored but now modelled, drop from the list: {sorted(stale)}"
    )
