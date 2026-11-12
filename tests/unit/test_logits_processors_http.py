"""A logits_processors value that HTTP cannot carry must be a 400 naming the field.

Processors are Python callables; a JSON list cannot be one, so any non-empty value was
accepted and then failed inside the engine (HTTP 200, empty content).
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "path,body",
    [
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
        ),
        ("/v1/completions", {"prompt": "hi"}),
    ],
)
def test_logits_processors_over_http_is_a_400_naming_the_field(client, path, body):
    body = {"model": "m", "max_tokens": 4, "logits_processors": [{"type": "noop"}], **body}
    r = client.post(path, json=body)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert "logits_processors" in err["message"]
