"""The synthetic replay cases are valid for our request models (so a failure on the server is the server's)."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts/research/agent_compat")
)
import synthetic_requests as sr  # noqa: E402

from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest  # noqa: E402
from yunshu_gateway.routers.chat import ChatCompletionRequest  # noqa: E402
from yunshu_gateway.routers.responses import ResponsesRequest  # noqa: E402

MODELS = {
    "/v1/chat/completions": ChatCompletionRequest,
    "/v1/messages": AnthropicMessagesRequest,
    "/v1/responses": ResponsesRequest,
}


def test_cases_parse():
    cs = sr.cases()
    assert len(cs) >= 8
    for c in cs:
        MODELS[c["path"]](model="m", **c["body"])
