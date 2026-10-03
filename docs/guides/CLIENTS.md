# Connecting clients

Yunshu speaks the OpenAI API under `/v1` and the Anthropic Messages API at `/v1/messages`.
Anything that lets you set a base URL can use it.

| | Value |
|---|---|
| OpenAI base URL | `http://127.0.0.1:8000/v1` |
| Anthropic base URL | `http://127.0.0.1:8000` (the SDK appends `/v1/messages`) |
| API key | any string, unless the server was started with `--auth-token` (then that token) |
| Model name | in single-model mode (`yunshu serve -m ...`) any name maps to the served model; in multi-model mode (`--models-dir`) use an id from `GET /v1/models` |

With `--auth-token`, the key is accepted as `Authorization: Bearer <token>` (OpenAI clients) or
`x-api-key: <token>` (Anthropic clients).

`yunshu serve` binds `127.0.0.1` by default. To reach it from another machine or from a Docker
container, start it with `--host 0.0.0.0 --auth-token <secret>`.

## curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "Hello"}]}'
```

## OpenAI Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
    # Passed to chat templates that support it (Qwen3.8: low / medium / xhigh).
    extra_body={"reasoning_effort": "medium"},
)
print(r.choices[0].message.content)
```

Streaming (`stream=True`), tools, `response_format` (JSON schema), `stop`, `logprobs` and `seed`
are supported; see [API.md](../API.md). Thinking output of reasoning models is streamed separately
from the answer (`delta.reasoning_content`).

## OpenAI JavaScript / TypeScript

```ts
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://127.0.0.1:8000/v1", apiKey: "local" });
const r = await client.chat.completions.create({
  model: "local",
  messages: [{ role: "user", content: "Hello" }],
});
console.log(r.choices[0].message.content);
```

## Anthropic Python

```python
import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:8000", api_key="local")
msg = client.messages.create(
    model="local",
    max_tokens=512,
    messages=[{"role": "user", "content": "Hello"}],
)
print(msg.content[-1].text)
```

## Coding agents

- **Claude Code** (Anthropic API):
  `ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=local claude`
- **Codex, OpenCode, Pi**: `yunshu launch <tool>` writes the tool's config to point at the running
  server (`yunshu launch list` shows what is installed).
- **Cline / Roo Code**: provider "OpenAI Compatible", base URL `http://127.0.0.1:8000/v1`, any key,
  model `local`.
- **Continue**: a model entry with `provider: openai`, `apiBase: http://127.0.0.1:8000/v1`,
  `model: local`.

`yunshu launch claude` also sets the model's real context window and adds `yunshu statusline` (prefill
progress, decode speed, last cache hit) to Claude Code's status line unless you have your own. Server-side
`web_search` / `web_fetch`, the MCP connector and what each agent supports are in
[AGENT_COMPAT.md](AGENT_COMPAT.md).

Agents send long, repeated prompts. Yunshu's prefix cache reuses the shared part, so after the first
turn only the new tokens are prefilled.

## Open WebUI

Settings → Connections → OpenAI API: URL `http://host.docker.internal:8000/v1` when Open WebUI runs
in Docker (start Yunshu with `--host 0.0.0.0 --auth-token <secret>` and use the secret as the key),
or `http://127.0.0.1:8000/v1` when it runs natively.

## Yunmo

Yunmo (a separate consumer project)'s `llm_adapter` is OpenAI-compatible: set its `base_url` to
`http://127.0.0.1:8000/v1`.

## Checking the server

| Endpoint | Auth | Use |
|---|---|---|
| `GET /health` | none | liveness + whether a model is loaded (`{"status": "ok", "engine": {"loaded": true}}`) |
| `GET /health/ready` | none | 200 when a model is loaded and memory is below 95%, else 503 (for load balancers and scripts) |
| `GET /version` | none | installed Yunshu version |
| `GET /v1/models` | token if set | served models |
| `GET /metrics` | token if set | Prometheus metrics |

`yunshu status` prints the same from the command line.
