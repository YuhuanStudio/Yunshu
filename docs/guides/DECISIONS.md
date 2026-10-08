# Typed decisions

Use `POST /v1/decisions` for predicate, choice and ordinal score questions. A decision
checkpoint evaluates the input and question schema in one forward pass; it generates
no text tokens. This is separate from chat completion and JSON-schema decoding.

## Load a checkpoint

Install the `vision` extra and serve a Clef MLX checkpoint, for example:

```bash
yunshu pull abenzerps/Clef-MLX
yunshu serve -m abenzerps/Clef-MLX
```

The merged loader supports the Cloudflare Clef joint schema head, including Clef-flash.
Other decision-head families (OpenJev, Laya and D1) are not supported by this loader.
Unknown heads fail to load; a chat model passed to the decision route returns 400.
See [model support](MODEL_SUPPORT.md) and [verification evidence](API_SURFACE.md).

## Ask a question

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
decision = client.decisions.create(
    model="abenzerps/Clef-MLX",
    input="The customer asks for a refund after a broken delivery.",
    questions=[{
        "type": "choice", "name": "route", "instructions": "Choose the support team.",
        "choices": [{"value": "support"}, {"value": "sales"}],
    }],
)
print(decision.answers)
```

Predicates return a probability; choices return the chosen typed value, confidence
and every option's probability. Scores return the expected zero-based level index,
confidence and level probabilities. Answers follow question order. A non-finite head
result becomes a refusal, never a fabricated probability. Usage has zero output tokens.

Inputs may be text or user messages with `input_text` and inline base64 `input_image`
data URLs. Remote image URLs are rejected; images require a vision-capable checkpoint.
The combined input/schema limit is 16,384 tokens; overflow is rejected, not truncated.
Requests are stateless and non-streaming. `safety_identifier` is accepted but ignored.

## System One wire format

`POST /v1/systemone` accepts `model`, `state`, a named `questions` map and optional
base64 `images`. Question types are `noul` (boolean), `choice` and `score`; `criteria`
defines choices or score levels. It uses the same decision engine and returns answers
keyed by question name. See [API reference](../API.md) for both wire formats.
