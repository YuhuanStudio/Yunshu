"""What each coding agent needs to know about a Yunshu model, derived from ``/v1/models``.

The agents cannot read a local server's model facts by themselves: Claude Code assumes a 200K
window for a model id it does not know, Codex has no catalog entry for it (so no reasoning-effort
choices and a fallback prompt), opencode wants ``limit.context``. The launcher fetches the model
card and hands each agent the numbers it would otherwise guess wrong. Nothing here touches the
network; ``ModelInfo.from_models_item`` takes the JSON ``/v1/models`` already returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A compact coding-agent prompt for Codex's ``base_instructions``. Codex's own default is ~17K characters
# written for frontier models; a local model prefills every character of it on each cold start.
CODEX_BASE_INSTRUCTIONS = """You are a coding agent running in the Codex CLI on the user's machine.
Work in the current directory. Read files before editing them, keep changes small and focused, and
run the project's tests or build after a change when you can.
Use the provided tools: run shell commands with exec_command, and edit files with the patch or shell
tools you were given. Prefer rg over grep for searching. Never guess file contents; look them up.
Explain what you did in a short final message: what changed and how you checked it.
If a request is ambiguous, pick the most reasonable reading and say what you assumed."""


@dataclass
class ModelInfo:
    id: str
    display_name: str = ""
    context: int = 131072
    max_output: int = 32768
    vision: bool = False
    tools: bool = True
    reasoning: bool = False
    effort_levels: list[str] = field(default_factory=list)
    default_effort: str | None = None
    web_search: bool = False
    web_search_provider: str | None = None

    @classmethod
    def from_models_item(cls, item: dict) -> ModelInfo:
        y = item.get("yunshu") or {}
        ctx = (
            item.get("max_input_tokens")
            or item.get("context_length")
            or (y.get("context") or {}).get("length")
        )
        out = item.get("max_tokens") or y.get("max_output_tokens")
        reasoning = y.get("reasoning") or {}
        arch = item.get("architecture") or {}
        ws = ((y.get("server_tools") or {}).get("web_search")) or {}
        levels = list(reasoning.get("effort_levels") or [])
        return cls(
            id=item["id"],
            display_name=item.get("display_name") or item["id"],
            context=int(ctx or 131072),
            max_output=int(out or 32768),
            vision="image" in (arch.get("input_modalities") or []),
            tools=bool((y.get("tools") or {}).get("supported", True))
            if isinstance(y.get("tools"), dict)
            else True,
            reasoning=bool(reasoning.get("supported")),
            effort_levels=levels,
            default_effort=reasoning.get("default_effort"),
            web_search=bool(ws.get("available")),
            web_search_provider=ws.get("provider"),
        )


# ── Claude Code ───────────────────────────────────────────────────────────────


def claude_code_env(
    info: ModelInfo, base_url: str, api_key: str = "", effort: str | None = None
) -> dict[str, str]:
    """Environment for ``claude``. Base URL has no ``/v1``: the SDK appends it.

    - ``ANTHROPIC_AUTH_TOKEN`` (bearer) instead of ``ANTHROPIC_API_KEY``: an API key makes the TUI ask once
      whether to trust it.
    - Every model alias (opus / sonnet / haiku / small-fast) resolves to the served model.
    - ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` is the real window: for an id it does not know Claude Code assumes
      200K and compacts too late (with it, ``/context`` shows the real window and auto-compact fires below it).
    - ``CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY`` fills ``/model`` from ``/v1/models``.
    """
    env = {
        "ANTHROPIC_BASE_URL": base_url.rstrip("/"),
        "ANTHROPIC_AUTH_TOKEN": api_key or "yunshu",
        "ANTHROPIC_MODEL": info.id,
        "ANTHROPIC_SMALL_FAST_MODEL": info.id,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": info.id,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": info.id,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": info.id,
        "CLAUDE_CODE_SUBAGENT_MODEL": info.id,
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(info.context),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(min(info.max_output, 32000)),
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "DISABLE_AUTOUPDATER": "1",
    }
    if effort:
        env["CLAUDE_CODE_EFFORT_LEVEL"] = effort
    return env


# ── Codex ─────────────────────────────────────────────────────────────────────


def codex_catalog(info: ModelInfo) -> dict:
    """A ``model_catalog_json`` entry: makes ``/model`` list the local model, with its real window and
    reasoning-effort choices (without one Codex prints "Model metadata not found" and sends no effort)."""
    levels = [
        {"effort": e, "description": _EFFORT_TEXT.get(e, e)} for e in info.effort_levels
    ]
    default = (
        info.default_effort
        if info.default_effort in info.effort_levels
        else (
            "medium"
            if "medium" in info.effort_levels
            else (info.effort_levels[0] if info.effort_levels else None)
        )
    )
    entry = {
        "slug": info.id,
        "display_name": info.display_name or info.id,
        "description": "Local model served by Yunshu",
        "supported_reasoning_levels": levels,
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 1,
        "base_instructions": CODEX_BASE_INSTRUCTIONS,
        "supports_reasoning_summaries": info.reasoning,
        "support_verbosity": False,
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_parallel_tool_calls": True,
        "context_window": info.context,
        "input_modalities": ["text", "image"] if info.vision else ["text"],
        "experimental_supported_tools": [],
    }
    if default:
        entry["default_reasoning_level"] = default
    return {"models": [entry]}


_EFFORT_TEXT = {
    "minimal": "Fastest, almost no reasoning",
    "low": "Fast responses with light reasoning",
    "medium": "Balanced speed and reasoning depth",
    "high": "Deeper reasoning",
    "xhigh": "Maximum reasoning depth",
}


def codex_provider_toml(
    info: ModelInfo, base_url: str, catalog_path: str, *, websockets: bool = False
) -> str:
    """The Codex ``config.toml`` keys and the ``[model_providers.yunshu]`` table (base URL includes ``/v1``)."""
    lines = [
        f'model = "{info.id}"',
        'model_provider = "yunshu"',
        f'model_catalog_json = "{catalog_path}"',
        f"model_context_window = {info.context}",
        f"model_auto_compact_token_limit = {int(info.context * 0.85)}",
        f'web_search = "{"live" if info.web_search else "disabled"}"',
    ]
    if info.reasoning:
        lines.append('model_reasoning_summary = "auto"')
    provider = [
        "",
        "[model_providers.yunshu]",
        'name = "Yunshu"',
        f'base_url = "{base_url.rstrip("/")}/v1"',
        'env_key = "YUNSHU_API_KEY"',
        'wire_api = "responses"',
    ]
    if websockets:
        provider.append("supports_websockets = true")
    return "\n".join(lines + provider) + "\n"


# ── opencode ──────────────────────────────────────────────────────────────────


def opencode_provider(info: ModelInfo, base_url: str, api_key: str = "") -> dict:
    """The ``provider.yunshu`` entry: ``limit`` gives opencode the context percentage and compaction point."""
    opts: dict = {"baseURL": base_url.rstrip("/") + "/v1"}
    if api_key:
        opts["apiKey"] = api_key
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Yunshu",
        "options": opts,
        "models": {
            info.id: {
                "name": info.display_name or info.id,
                "tool_call": info.tools,
                "reasoning": info.reasoning,
                "limit": {"context": info.context, "output": info.max_output},
                "modalities": {
                    "input": ["text", "image"] if info.vision else ["text"],
                    "output": ["text"],
                },
            }
        },
    }
