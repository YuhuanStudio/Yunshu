"""Image-input checks of the route registry (imported at the bottom of route_checks.py): a vision
model reads an image through all three dialects, a text-only model refuses with a 400 that says so,
and a degenerate image is a client error, never a 500."""

from __future__ import annotations

import base64

from route_checks import Ctx, _png, check, err_ok, expect


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


@check(
    "vision_input",
    "POST /v1/chat/completions",
    "POST /v1/messages",
    "POST /v1/responses",
    "POST /api/chat",
)
def _vision_input(c: Ctx):
    png = _png(224)
    chat_msg = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What colour is this image? One word."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + _b64(png)},
                },
            ],
        }
    ]
    if c.kind != "vlm":
        # a text-only model refuses in every dialect with the reason, in the dialect's shape
        r = c.req(
            "POST",
            "/v1/chat/completions",
            json={"model": c.model, "messages": chat_msg, "max_tokens": 8},
        )
        err_ok(r, "openai")
        expect(r.status_code == 400, f"chat image on a text model -> {r.status_code}")
        expect(
            "image" in r.json()["error"]["message"].lower(),
            f"message {r.json()['error']['message']!r}",
        )
        r = c.req(
            "POST",
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": c.model,
                "max_tokens": 8,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _b64(png),
                                },
                            },
                            {"type": "text", "text": "What colour?"},
                        ],
                    }
                ],
            },
        )
        err_ok(r, "anthropic")
        expect(
            r.status_code == 400, f"messages image on a text model -> {r.status_code}"
        )
        return
    base = c.oa.chat.completions.create(
        model=c.model,
        messages=[{"role": "user", "content": "What colour is this image? One word."}],
        max_tokens=4,
    )
    r = c.oa.chat.completions.create(model=c.model, messages=chat_msg, max_tokens=24)
    expect(r.choices and r.choices[0].message, "chat with an image")
    expect(
        r.usage.prompt_tokens > base.usage.prompt_tokens + 10,
        f"image not counted: {r.usage.prompt_tokens} vs {base.usage.prompt_tokens}",
    )
    m = c.an.messages.create(
        model=c.model,
        max_tokens=24,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _b64(png),
                        },
                    },
                    {"type": "text", "text": "What colour is this image? One word."},
                ],
            }
        ],
    )
    expect(
        m.usage.input_tokens > base.usage.prompt_tokens + 10,
        "messages image not counted",
    )
    o = c.oa.responses.create(
        model=c.model,
        max_output_tokens=24,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "What colour is this image? One word.",
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64," + _b64(png),
                    },
                ],
            }
        ],
    )
    expect(
        o.usage.input_tokens > base.usage.prompt_tokens + 10,
        "responses image not counted",
    )
    ol = c.req(
        "POST",
        "/api/chat",
        json={
            "model": c.model,
            "stream": False,
            "options": {"num_predict": 16},
            "messages": [
                {
                    "role": "user",
                    "content": "What colour is this image?",
                    "images": [_b64(png)],
                }
            ],
        },
    )
    expect(
        ol.status_code == 200
        and ol.json().get("prompt_eval_count", 0) > base.usage.prompt_tokens + 10,
        f"ollama images {ol.status_code} {ol.text[:150]}",
    )
    # a degenerate (1x1) image: the answer is a client error with a message or a normal reply,
    # never a bare 500 (the processor cannot patch a 1-pixel image)
    tiny = _png(1)
    t = c.req(
        "POST",
        "/v1/chat/completions",
        json={
            "model": c.model,
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + _b64(tiny)},
                        },
                    ],
                }
            ],
        },
    )
    c.notes["tiny_image"] = f"{t.status_code}: {t.text[:100]}"
    expect(t.status_code != 500, f"1x1 image -> bare 500: {t.text[:200]}")
    if t.status_code >= 400:
        err_ok(t, "openai")
