# Coding-agent compatibility

Yunshu's goal for coding agents is near-native fidelity: Claude Code, Codex CLI, opencode and the SDKs
should find every endpoint, header, beta, tool type and server-side tool they use working like the real
Anthropic / OpenAI APIs, not only correct generation. This page is the evidence: what each agent
actually calls (a census of the real CLIs), what Yunshu does with it, and what is still missing.

The raw census data is private (`docs/research/runs/<date>-agent-census/`, gitignored); the tools that
produce it are in `scripts/research/agent_compat/` (below).

## Method

| Piece | What it does |
|---|---|
| `census_server.py` | A scripted, recording mock model server. Records **every** request (method, path, query, headers, body), answers a script of tool calls so the agent walks a session, returns 404 for unknown paths, and answers `web_search_*` requests with spec-shaped server-tool blocks so the client parser is exercised. No GPU. |
| `census.py` | Runs the pinned CLIs (Claude Code 2.1.285, Codex 0.157.1, opencode 1.18.33 from `agentic-clis`) through 27 scripted sessions: plain, bash / edit, sub-agents, web search / fetch, MCP, images, thinking, `/context`, `/cost`, `/compact`, status line, model discovery, Codex reasoning / compaction / websocket / catalog, opencode. Isolation: scrubbed env, isolated `HOME` / `CLAUDE_CONFIG_DIR` / `CODEX_HOME`, `sandbox-exec` denying everything but loopback. |
| `census_tui.py` | Drives an agent's interactive TUI in a pty (`/status`, `/model`). |
| `replay.py`, `replay_report.py` | Replays every recorded request against a real Yunshu (baseline vs current) and compares status / stream validity. |
| `e2e_agents.py`, `e2e_sdk.py`, `capture_proxy.py`, `fake_searxng.py`, `tiny_mcp.py` | End-to-end runs of the real agents and the official SDKs against a real model, with a fake SearXNG (fixed corpus with a checkable fact), a tiny MCP server and a full-capture proxy. |

## What the agents call

### Claude Code 2.1.285

| Request | When | Yunshu |
|---|---|---|
| `POST /v1/messages?beta=true`, `stream: true` | every turn. `anthropic-beta: claude-code-20250219,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,mid-conversation-tool-changes-2026-07-01,effort-2025-11-24`. Body: `system` as 3 blocks (`cache_control: ephemeral` on blocks 1-2, block 0 is `x-anthropic-billing-header: ...`), `messages` including `role: "system"` entries mid-conversation, 18-20 client tools, `thinking: {type: "adaptive"}`, `output_config: {effort}`, `context_management: {edits: [clear_thinking_20251015]}`, `metadata.user_id`, `max_tokens: 32000` | served. `adaptive` thinking and `output_config` used to be a 422 (`thinking.type must be 'enabled' or 'disabled'`); fixed. |
| `POST /v1/messages/count_tokens?beta=true` | `/context` (14 calls: one per tool group and message category), beta `token-counting-2024-11-01` | served (counts system, messages, tools) |
| `GET /v1/models?limit=1000` | only with `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`: fills `/model` | served, with `max_input_tokens`, `max_tokens`, `capabilities` |
| `POST /v1/messages` with `tools: [{type: "web_search_20250305", name: "web_search", max_uses: 8}]`, `tool_choice: auto`, the prompt "Perform a web search for the query: ..." | its WebSearch tool: the search runs **server-side** and the client renders the `web_search_tool_result` blocks | implemented (server tools, below) |
| small-model calls (`ANTHROPIC_SMALL_FAST_MODEL`, Haiku alias) | titles, summaries, the WebFetch page summary | served |
| MCP servers | **client-side**: the CLI connects, tools reach the model as `mcp__server__tool` function tools | nothing to serve; needs ordinary tool calling |
| WebFetch | **client-side** fetch, then a small-model summary call | nothing to serve |
| sub-agents (`Agent` / `Task`), `/compact` | ordinary `/v1/messages` calls | served |

With traffic to Anthropic blocked, nothing else was requested: no telemetry, OAuth or update calls. The
interactive TUI runs a preflight against `api.anthropic.com/api/hello` only during first-run onboarding.

Facts the CLI does **not** learn from a custom server: the context window (it assumes 200K for an unknown
model id, which is why `CLAUDE_CODE_MAX_CONTEXT_TOKENS` exists; `/v1/models` `max_input_tokens` feeds the
picker only) and prices (it prints a made-up cost for the unknown model). `yunshu launch claude` sets the
window; the cost line is Claude Code's own arithmetic and cannot be corrected from the server.

### Codex CLI 0.157.1

| Request | When | Yunshu |
|---|---|---|
| `POST /v1/responses`, `stream: true`, `store: false` | every turn. Body: `instructions` (~17 KB), `input` items (`developer` and `user` messages with `input_text`, `function_call`, `function_call_output`, `reasoning`), `tools` of type `function`, `namespace` (`multi_agent_v1`, with nested `tools`) and `web_search`, `parallel_tool_calls`, `reasoning: {effort?, summary: "auto"}`, `include: ["reasoning.encrypted_content"]`, `prompt_cache_key`, `client_metadata`. Headers `originator: codex_exec`, `session-id`, `thread-id`, `x-codex-*` | served; the `namespace` and `web_search` tools used to be a 422 (the tool model required a `name`); fixed |
| `GET /v1/responses` with `Upgrade: websocket`, `OpenAI-Beta: responses_websockets=2026-02-06` | only when the provider sets `supports_websockets = true`; three attempts, then POST | `WS /v1/responses` exists (see TRANSPORTS.md) |
| compaction | local: an ordinary `POST /v1/responses` with `tools: []`, `parallel_tool_calls: false` and the summary prompt; **no** `/responses/compact` for a custom provider | served |
| model metadata | **never** fetched from a custom provider (no `GET /models`); without a catalog Codex prints "Model metadata not found", sends no reasoning effort and offers only its built-in OpenAI models in `/model` | `yunshu launch codex` writes `model_catalog_json` |
| `web_search` (`web_search = "live"`) | adds a `{type: "web_search"}` tool; expects `web_search_call` items | implemented |
| MCP servers | client-side (function tools) | nothing to serve |

### opencode 1.18.33

| Request | When | Yunshu |
|---|---|---|
| `POST /v1/chat/completions`, `stream_options: {include_usage: true}`, `max_tokens: 32000` | every turn; 9 function tools; title generation runs a first request with `reasoning_effort: "low"`; headers `x-session-id`, `x-session-affinity` | served |
| `/models` | not fetched (`OPENCODE_DISABLE_MODELS_FETCH`); limits come from `provider.*.models.*.limit` | `yunshu launch opencode` writes them |

## Feature matrix

Status: **works** (verified, evidence named), **partial**, **missing**. "E2E" = the real agent against a real
Yunshu; "SDK" = the official SDK; "unit" = `tests/unit` with fakes.

| Feature | Claude Code | Codex | opencode | SDK users | Evidence |
|---|---|---|---|---|---|
| Streaming generation, tool calls | works | works | works | works | census replay, E2E |
| Thinking / reasoning blocks in the native shape | works (`thinking` blocks; `adaptive` accepted) | works (`reasoning` items with `summary`, `encrypted_content` round-trip) | works (`reasoning_content`) | works | unit |
| Effort control | works (`output_config.effort` -> template `reasoning_effort`) | works (`reasoning.effort` via catalog levels) | works (`reasoning_effort`) | works | unit, launch |
| Web search, server-side | see E2E results | see E2E results | not used (client tool) | works | unit, E2E |
| Web fetch, server-side | client-side in the CLI | not used | not used | works | unit, E2E |
| MCP connector (`mcp_servers`, `{type: "mcp"}`) | not used (client-side MCP) | not used | not used | works | unit, E2E |
| Model discovery / picker | works with launch env | works with catalog | n/a | works | census |
| Context window / auto-compact point | works with launch env | works with catalog | works with `limit` | works | launch |
| `count_tokens` | works | n/a | n/a | works | census replay |
| Prefix-cache reuse on agent traffic | see Known gaps | | | | |

## Server-side tools

Off unless configured. A request that asks for web search with no provider gets the API's own error
(`web_search_tool_result_error`, `error_code: "unavailable"` / a failed `web_search_call`), the model is told
why, and `x_yunshu.server_tools` carries the setup hint. `GET /v1/models` advertises the state under
`yunshu.server_tools`.

| Setting | Meaning |
|---|---|
| `YUNSHU_SEARXNG_URL` | a self-hosted SearXNG with the JSON format enabled (the private default recommendation) |
| `YUNSHU_BRAVE_API_KEY`, `YUNSHU_TAVILY_API_KEY`, `YUNSHU_EXA_API_KEY` | hosted providers; `YUNSHU_WEB_SEARCH_PROVIDER` picks one (`auto` prefers SearXNG) |
| `YUNSHU_WEB_FETCH`, `YUNSHU_WEB_FETCH_ALLOW_PRIVATE`, `..._MAX_BYTES`, `..._TIMEOUT` | web_fetch needs no provider; private, loopback and link-local addresses are blocked (also after redirects and DNS), 2 MB and 20 s by default |
| `YUNSHU_MCP_CONNECTOR`, `YUNSHU_MCP_CONNECTOR_ALLOW_PRIVATE`, `..._TIMEOUT` | the MCP connector (streamable HTTP, legacy SSE) |
| `YUNSHU_SERVER_TOOL_MAX_ITERATIONS` | generate / run / continue rounds per request (8) |

See [CONFIGURATION.md](../CONFIGURATION.md) for the full table.

## Launching an agent

`yunshu launch claude|codex|opencode` reads the model card from `/v1/models` and hands the agent what it
would otherwise guess wrong; `--dry-run` prints it.

| Agent | What is set |
|---|---|
| Claude Code | `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, every model alias -> the served model, `CLAUDE_CODE_MAX_CONTEXT_TOKENS` (the real window), `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY`, optional `CLAUDE_CODE_EFFORT_LEVEL` (`--effort`). Environment only: `~/.claude` is not touched. |
| Codex | `~/.codex/config.toml` (`model_provider`, `model_context_window`, `model_auto_compact_token_limit`, `web_search = "live"` when a provider is configured) and `~/.codex/yunshu-models.json` (`model_catalog_json`: window, reasoning levels, modalities, a compact `base_instructions`). |
| opencode | `provider.yunshu` in `opencode.json` with `limit.context` / `limit.output`, `reasoning`, `tool_call`, `modalities`. |
