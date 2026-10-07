import { t } from "./i18n/index.ts";

/**
 * Setup commands for common clients, generated for the current service address
 * and model. Shapes follow docs/guides/CLIENTS.md and AGENT_COMPAT.md; the real
 * access token is never embedded, only an environment variable reference.
 */
export interface Integration {
  id: string;
  title: string;
  description: string;
  language: string;
  filename?: string;
  code: string;
}

export const serviceRoot = (baseUrl: string) =>
  baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");

export function buildIntegrations(
  baseUrl: string,
  model: string,
): Integration[] {
  const root = serviceRoot(baseUrl);
  const v1 = `${root}/v1`;
  const m = model || "local";
  const key = '"${YUNSHU_AUTH_TOKEN:-local}"';
  return [
    {
      id: "claude-code",
      title: "Claude Code",
      description: t("api.integration.claudeCode.description"),
      language: "bash",
      code: [
        `export ANTHROPIC_BASE_URL=${root}`,
        `export ANTHROPIC_AUTH_TOKEN=${key}`,
        `export ANTHROPIC_MODEL=${m}`,
        `export ANTHROPIC_DEFAULT_HAIKU_MODEL=${m}`,
        `export ANTHROPIC_DEFAULT_SONNET_MODEL=${m}`,
        `export ANTHROPIC_DEFAULT_OPUS_MODEL=${m}`,
        "claude",
      ].join("\n"),
    },
    {
      id: "codex",
      title: "Codex",
      description: t("api.integration.codex.description"),
      language: "toml",
      filename: "config.toml",
      code: [
        `model = "${m}"`,
        'model_provider = "yunshu"',
        "",
        "[model_providers.yunshu]",
        'name = "Yunshu"',
        `base_url = "${v1}"`,
        'env_key = "YUNSHU_API_KEY"',
        'wire_api = "responses"',
      ].join("\n"),
    },
    {
      id: "opencode",
      title: "opencode",
      description: t("api.integration.opencode.description"),
      language: "json",
      filename: "opencode.json",
      code: JSON.stringify(
        {
          $schema: "https://opencode.ai/config.json",
          model: `yunshu/${m}`,
          provider: {
            yunshu: {
              npm: "@ai-sdk/openai-compatible",
              name: "Yunshu",
              options: { baseURL: v1, apiKey: "{env:YUNSHU_AUTH_TOKEN}" },
              models: { [m]: { name: m, tool_call: true, reasoning: true } },
            },
          },
        },
        null,
        2,
      ),
    },
    {
      id: "openai",
      title: "OpenAI Python SDK",
      description: t("api.integration.openai.description"),
      language: "python",
      code: [
        "import os",
        "from openai import OpenAI",
        "",
        `client = OpenAI(base_url=${JSON.stringify(v1)}, api_key=os.environ.get("YUNSHU_AUTH_TOKEN", "local"))`,
        "reply = client.chat.completions.create(",
        `    model=${JSON.stringify(m)},`,
        '    messages=[{"role": "user", "content": "Hello"}],',
        ")",
        "print(reply.choices[0].message.content)",
      ].join("\n"),
    },
    {
      id: "anthropic",
      title: "Anthropic SDK",
      description: t("api.integration.anthropic.description"),
      language: "python",
      code: [
        "import os",
        "import anthropic",
        "",
        `client = anthropic.Anthropic(base_url=${JSON.stringify(root)}, api_key=os.environ.get("YUNSHU_AUTH_TOKEN", "local"))`,
        "msg = client.messages.create(",
        `    model=${JSON.stringify(m)},`,
        "    max_tokens=512,",
        '    messages=[{"role": "user", "content": "Hello"}],',
        ")",
        "print(msg.content[-1].text)",
      ].join("\n"),
    },
    {
      id: "curl",
      title: "curl",
      description: t("api.integration.curl.description"),
      language: "bash",
      code: [
        `curl ${v1}/chat/completions \\`,
        `  -H "Authorization: Bearer ${"${YUNSHU_AUTH_TOKEN:-local}"}" \\`,
        '  -H "Content-Type: application/json" \\',
        `  -d '${JSON.stringify({ model: m, messages: [{ role: "user", content: "Hello" }] })}'`,
      ].join("\n"),
    },
  ];
}
