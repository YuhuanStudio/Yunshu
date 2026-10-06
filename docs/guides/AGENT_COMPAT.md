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

## Drift against the latest CLIs

The tables above describe the pinned census CLIs. `scripts/dev/agentcompat` (below) reruns the census with
the newest releases and fails when anything new is not written down here. Last run: Claude Code 2.1.291,
Codex 0.160.1, opencode 1.18.34.

| New item | Seen in | Yunshu |
|---|---|---|
| `anthropic-beta: thinking-display-updates-2026-08-18` | Claude Code 2.1.291, every turn | accepted; betas are never rejected |
| `thinking: {type: "adaptive", display: "updates"}` | Claude Code 2.1.291 | accepted (the `thinking` object is read for `type` / `budget_tokens`); `display` has no local meaning and is ignored |
| header `x-opencode-session-id` | opencode 1.18.34 | ignored (`x-session-id` already feeds the session affinity) |

## Verifying the claims: `scripts/dev/agentcompat`

Documents drift, so the claims above are re-earned by one command with one verdict
(`docs/research/runs/<date>-agentcompat-<commit>/verdict.json`, exit 0 only on positive evidence):

| Stage | What it proves |
|---|---|
| `install` | installs the newest Claude Code / Codex / opencode under `/Volumes/P5Plus/yunshu-build/agentic-clis-latest` (never global; `install_agents.sh` with `*_V=latest`) |
| `census` | the scripted sessions with those CLIs against the recording mock; diff against the pinned census (paths, query keys, headers, beta values, body fields, tool / block types); every new item must be named in this file |
| `m3` | a gpuq M3 job (`m3_serve.py`, Qwen3.5-9B-4bit, loopback only) behind an `ssh -L` forward; every recorded census request is replayed (status 2xx, SSE event order and fields, body validated by the official `anthropic` / `openai` models), then the scripted agent tasks run with the latest CLIs; the job is stopped over a loopback control port and the tunnel killed |

`python/` spec drift is covered by a unit test (`test_spec_field_inventory.py`): every parameter of the installed
`openai` / `anthropic` SDK request types is either a field of our request model or listed with a reason, so a new
SDK release that adds a field fails CI instead of being silently dropped. Run as an extra stage of `m3sweep` with
`scripts/dev/agentcompat --stages m3`.

## Feature matrix

Status: **works** (verified, evidence named), **partial**, **missing**, **n/a** (the agent never asks for it).
"E2E" = the real pinned agent against a real Yunshu (Qwen3.5-9B and Qwen3.8-27B-oQ4e-mtp, launch-helper
configuration, capture proxy); "SDK" = the official SDK against a real server; "unit" = `tests/unit` with fakes.
Last full rerun: current main plus the fixes listed under "Found and fixed by the last rerun".

| Feature | Claude Code | Codex | opencode | SDK users | Evidence |
|---|---|---|---|---|---|
| Streaming generation, tool calls | works | works | works | works | E2E edit / bash tasks in all three, unit |
| Mid-conversation system messages (per-turn notes) | works (text and image turns) | works (`developer` items) | n/a | works | unit; the image-turn case was broken, see below |
| Thinking / reasoning blocks in the native shape | works (`thinking` blocks, `adaptive` accepted) | works (`reasoning` items, `encrypted_content` round-trip) | works (`reasoning_content`) | works | unit, SDK |
| Effort control | works (`output_config.effort` -> template `reasoning_effort`) | works (`reasoning.effort` via catalog levels) | works (`reasoning_effort`) | works | unit, launch |
| Web search, server-side | works (WebSearch tool, fake SearXNG fact found) | works (`web_search`, fact found) | n/a (client tool) | works (Messages, Responses, streamed and not) | E2E, SDK |
| Web fetch | client-side in the CLI | n/a | n/a | works (server-side, public page and SSRF-guarded loopback) | SDK |
| MCP | client-side, works (tiny MCP server, answer 42) | client-side, works (answer 42) | n/a | server-side connector works (Messages, Responses) | E2E, SDK |
| Images (tool result / input) | works after the fix (Read of a PNG, answer "red") | n/a in exec | n/a | works | E2E |
| Model discovery / picker | works (`/model` lists the served model, launch env) | works (catalog: `/model` lists it with its description) | n/a | works | TUI capture, census |
| Context window / auto-compact point | works (`/context` 262.1k window, 33k autocompact buffer) | works (catalog `context_window`) | works (`limit.context`) | works | E2E `/context`, launch |
| Cost display | shows `$0.0000` in `/cost` (real: nothing is billed); the per-turn `total_cost_usd` is Claude Code's own arithmetic for an unknown model and cannot be changed server-side | n/a | n/a | n/a | TUI capture |
| Rate-limit headers | n/a: no agent needs them for an API-key / custom provider; Yunshu has no quota, so it reports none (never invented) | n/a | n/a | n/a | census (no agent read them) |
| `count_tokens` | works | n/a | n/a | works | census replay |
| Usage fields | works (`input_tokens`, `cache_read_input_tokens`, `output_tokens`, thinking tokens) | works | works (`include_usage`) | works | E2E capture |
| Engine failure mid-request | works: an SSE `error` event with the engine's message (was an empty successful reply) | works: `response.failed` with the message | works: an error chunk | works | unit |
| Websocket transport | n/a | works (`generate:false` prewarm, `WS /v1/responses`) | n/a | n/a | E2E `cx_ws` |
| Live engine state in the agent UI (Yunshu extra) | works: `yunshu launch claude` installs `yunshu statusline` (prefill %, decode tok/s, last cache hit, ctx %) unless the user has their own status line | not available (Codex has no status-line hook that runs a command) | not available | `x_yunshu` fields, `: yunshu-progress` SSE comments | unit |
| Prefix-cache reuse on agent traffic | works: per-request `x_yunshu.cache` and `X-Yunshu-Cache-*` | works | works | works | APC audit (see PERF_TREND) |

### Found and fixed by the last rerun

| Symptom | Cause | Fix |
|---|---|---|
| Claude Code turn after `Read` of an image: empty reply in 20 ms | The vision path skipped the family message adapter, so Qwen's template raised "System message must be at the beginning" on Claude Code's per-turn system note; the failure was then reported as a normal empty `end_turn` | The vision path runs the adapter; a template / engine error after the stream started is now an `error` event on Messages, chat and Responses |
| `/status`, `/model` capture of Claude Code stopped at the welcome screen | With `CLAUDE_CONFIG_DIR` set, the interactive UI reads its onboarding state from inside that directory | The census harness writes it there; screens are rendered through a terminal emulator |

### Sampled speculative decoding

Sampled requests (temperature > 0, what every coding agent sends) draft through the MTP lane with position-keyed
Gumbel sampling (`keyed_sampling.py`). The first version produced stray multilingual tokens and repetition on the
27B agent scenarios. Root cause: the noise took 24 random bits, whose top bucket is `1 - 2^-25`; float32 rounds it
to 1.0, so `-log(-log(u))` is `+inf`. About 1.5% of positions had a token with infinite noise, which won even
when top-p / top-k had filtered it out (or was NaN against its `-inf` row): a random token from the 248K vocabulary.
Fixed with 23-bit noise and by masking filtered logits after adding the noise. Greedy requests never hit it.

Re-verified on Qwen3.8-27B-oQ4e-mtp: the lane is token-identical to keyed serial draws (the same lane with every
draft rejected) on 16 prompt / sampling / seed combinations (T 0.6 / 0.7 / 1.0, top-p, top-k, min-p), the draws
match softmax(filtered logits / T), none fall outside the filtered support, and `cc_image`, `cx_mcp`, `oc_bash`,
`oc_edit`, `cc_edit`, `cx_edit`, `cc_mcp` all pass with clean output.

### Known limits (not Yunshu defects)

- Claude Code's `/model` still lists its built-in "Fable" row, and the default row carries a `[1m]` label: both come from the
  client's own catalog. The window it works with is the real one (`/context`).
- The non-streaming Responses `web_search` SDK check is model-dependent: a small model sometimes answers without calling
  the tool (same inner request as the streamed one, which searched). The server-side loop is the same code for both.
- Codex `/status` needs a completed turn before it prints token usage; the capture only shows the header and `/model`.

## Server-side tools

Off unless configured. A request that asks for web search with no provider gets the API's own error
(`web_search_tool_result_error`, `error_code: "unavailable"` / a failed `web_search_call`), the model is told
why, and `x_yunshu.server_tools` carries the setup hint. `GET /v1/models` advertises the state under
`yunshu.server_tools`.

| Setting | Meaning |
|---|---|
| `YUNSHU_SEARXNG_URL` | a self-hosted SearXNG with the JSON format enabled (the private default recommendation) |
| `YUNSHU_BRAVE_API_KEY`, `YUNSHU_TAVILY_API_KEY`, `YUNSHU_EXA_API_KEY` | hosted providers; `YUNSHU_WEB_SEARCH_PROVIDER` picks one (`auto` prefers SearXNG) |
| `YUNSHU_WEB_FETCH`, `YUNSHU_WEB_FETCH_ALLOW_PRIVATE`, `..._MAX_BYTES`, `..._TIMEOUT` | web_fetch needs no provider; private, loopback and link-local addresses are blocked (also after redirects and DNS), 2 MB and 20 s by default | Model input is parsed tolerantly (`uri`/`link`/`href` keys, a bare string, nested `input`, truncated JSON, a scheme-less host) and an unusable call returns an `invalid_tool_input` error that states the expected `{"url": "https://..."}`.
| `YUNSHU_MCP_CONNECTOR`, `YUNSHU_MCP_CONNECTOR_ALLOW_PRIVATE`, `..._TIMEOUT` | the MCP connector (streamable HTTP, legacy SSE) |
| `YUNSHU_SERVER_TOOL_MAX_ITERATIONS` | generate / run / continue rounds per request (8) |

See [CONFIGURATION.md](../CONFIGURATION.md) for the full table.

## Launching an agent

`yunshu launch claude|codex|opencode` reads the model card from `/v1/models` and hands the agent what it
would otherwise guess wrong; `--dry-run` prints it.

| Agent | What is set |
|---|---|
| Claude Code | `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, every model alias -> the served model, `CLAUDE_CODE_MAX_CONTEXT_TOKENS` (the real window), `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY`, optional `CLAUDE_CODE_EFFORT_LEVEL` (`--effort`), and a `--settings` status line running `yunshu statusline` (live prefill progress, decode speed, last cache hit, context %; skipped when you have your own `statusLine`, off with `--no-statusline`). Environment and flags only: `~/.claude` is not written. |
| Codex | `~/.codex/config.toml` (`model_provider`, `model_context_window`, `model_auto_compact_token_limit`, `web_search = "live"` when a provider is configured) and `~/.codex/yunshu-models.json` (`model_catalog_json`: window, reasoning levels, modalities, a compact `base_instructions`). |
| opencode | `provider.yunshu` in `opencode.json` with `limit.context` / `limit.output`, `reasoning`, `tool_call`, `modalities`. |

The VLM runner maps Claude `cache_control` markers through conversion and the
actual chat template, including native tool definitions and message text. It
reports actual checkpoint writes and reads rather than estimating tokens from
system text. Repeated cold prefixes can wait on an admitted producer's planned
hybrid checkpoint. Waiters stay outside the generator until publication is
attempted, then use the ordinary APC lookup; cancellation or failure releases
waiters to cold fallback. Media salts stay part of identity and each request
keeps its own mutable KV, sampler and detokenizer. Uniform clients in one
upstream generator may already avoid duplicate prefill; cross-generator groups
are measured separately in the single-flight replay.

System-first templates receive one leading instruction block. Anthropic top-level
`system` and system reminders are merged in their input order; OpenAI system and
developer messages are likewise hoisted. Reminders retain instruction priority.
Changing a reminder changes the leading prompt and can reduce prefix reuse. Renaming
a checkpoint does not disable this normalization for an otherwise unknown family.
