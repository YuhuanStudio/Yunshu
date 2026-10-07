import type zh from "../zh-TW/api.ts";
import type { Shape } from "../../types.ts";

const api: Shape<typeof zh> = {
  "address.anthropic": "Anthropic base URL",
  "address.description":
    "Paste into an SDK or coding agent as the base URL. Example commands use the model selected below.",
  "address.modelAria": "Model for integrations",
  "address.modelLabel": "Model used in commands",
  "address.openai": "OpenAI API base URL",
  "address.title": "Server address",
  "catalog.copy": "Copy full URL of {method} {path}",
  "catalog.description":
    "Read from this engine's OpenAPI schema: {count, plural, one {# operation} other {# operations}}.",
  "catalog.docs": "API docs",
  "catalog.empty": "No matching APIs",
  "catalog.emptyNote":
    "This catalog reflects the routes the server actually serves; nothing is inferred.",
  "catalog.error": "Could not load the API schema",
  "catalog.loading": "Loading API schema…",
  "catalog.search.aria": "Search API",
  "catalog.search.placeholder": "Search path, method or function",
  "catalog.table.aria": "Server API catalog",
  "catalog.table.function": "Function",
  "catalog.table.method": "Method",
  "catalog.table.path": "Path",
  "catalog.tag.all": "All",
  "catalog.tag.aria": "API category",
  "catalog.tag.other": "Other",
  "catalog.title": "Full API of this server",
  "catalog.unavailable": "API schema unavailable",
  "clients.note":
    "Commands only reference the {env} environment variable ({fallback} when unset). A real token is never written into them.",
  "clients.title": "Client setup",
  "integration.anthropic.description":
    "base_url excludes /v1; the SDK adds the path itself.",
  "integration.claudeCode.description":
    "Uses the Anthropic Messages API; every model alias points to the current model. You can also run yunshu launch claude to set it up.",
  "integration.codex.description":
    "Add to ~/.codex/config.toml. Uses the Responses API and the YUNSHU_API_KEY environment variable. You can also run yunshu launch codex.",
  "integration.curl.description": "Call chat completions directly.",
  "integration.openai.description":
    "The standard OpenAI API; just change base_url.",
  "integration.opencode.description":
    "Add to opencode.json. Adjust limit to the model's actual context length. You can also run yunshu launch opencode.",
  "page.description":
    "Connect your app to local models with the SDKs and coding agents you already use.",
  "page.title": "API access",
};
export default api;
