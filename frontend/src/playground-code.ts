import {
  buildPayload,
  endpointUrl,
  type CompletionBody,
  type Dialect,
} from "./stream.ts";

export type CodeLanguage = "curl" | "python" | "javascript";

export const DIALECT_LABEL: Record<Dialect, string> = {
  chat: "Chat Completions",
  responses: "Responses",
  messages: "Anthropic Messages",
};

const MAX_INLINE = 160;

/** Long image data URLs would drown the snippet; shorten them and say so. */
function shorten(value: unknown): { value: unknown; shortened: boolean } {
  let shortened = false;
  const walk = (v: unknown): unknown => {
    if (typeof v === "string") {
      if (v.length > MAX_INLINE && /^data:|^[A-Za-z0-9+/=]{200,}$/.test(v)) {
        shortened = true;
        return `${v.slice(0, 48)}…<${v.length} chars omitted>`;
      }
      return v;
    }
    if (Array.isArray(v)) return v.map(walk);
    if (v && typeof v === "object")
      return Object.fromEntries(
        Object.entries(v).map(([k, x]) => [k, walk(x)]),
      );
    return v;
  };
  return { value: walk(value), shortened };
}

function pyLiteral(v: unknown, indent = 0): string {
  const pad = "    ".repeat(indent + 1);
  const end = "    ".repeat(indent);
  if (v === null || v === undefined) return "None";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") return String(v);
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) {
    if (!v.length) return "[]";
    return `[\n${v.map((x) => `${pad}${pyLiteral(x, indent + 1)},`).join("\n")}\n${end}]`;
  }
  const entries = Object.entries(v as Record<string, unknown>);
  if (!entries.length) return "{}";
  return `{\n${entries
    .map(([k, x]) => `${pad}${JSON.stringify(k)}: ${pyLiteral(x, indent + 1)},`)
    .join("\n")}\n${end}}`;
}

export interface CodeRequest {
  dialect: Dialect;
  baseUrl: string;
  body: CompletionBody;
}

export interface CodeSnippets {
  url: string;
  snippets: Record<CodeLanguage, string>;
  /** True when an image payload was shortened for display. */
  shortened: boolean;
}

/** Snippets that send exactly what the Playground sends for this request. */
export function buildSnippets({
  dialect,
  baseUrl,
  body,
}: CodeRequest): CodeSnippets {
  let url: string;
  try {
    url = endpointUrl(baseUrl, dialect);
  } catch {
    url = baseUrl;
  }
  const { value, shortened } = shorten(buildPayload(dialect, body));
  const json = JSON.stringify(value, null, 2);
  const curl = [
    `curl -N ${shellQuote(url)} \\`,
    `  -H "Authorization: Bearer $YUNSHU_API_KEY" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d ${shellQuote(json)}`,
  ].join("\n");
  const python = [
    "import json",
    "import os",
    "",
    "import requests",
    "",
    `payload = ${pyLiteral(value)}`,
    "",
    "with requests.post(",
    `    ${JSON.stringify(url)},`,
    '    headers={"Authorization": f"Bearer {os.environ[\'YUNSHU_API_KEY\']}"},',
    "    json=payload,",
    "    stream=True,",
    ") as response:",
    "    response.raise_for_status()",
    "    for line in response.iter_lines(decode_unicode=True):",
    '        if line and line.startswith("data: "):',
    "            print(line[6:])",
  ].join("\n");
  const javascript = [
    `const response = await fetch(${JSON.stringify(url)}, {`,
    '  method: "POST",',
    "  headers: {",
    "    Authorization: `Bearer ${process.env.YUNSHU_API_KEY}`,",
    '    "Content-Type": "application/json",',
    "  },",
    `  body: JSON.stringify(${indent(json, 2)}),`,
    "});",
    "if (!response.ok) throw new Error(await response.text());",
    "const decoder = new TextDecoder();",
    "for await (const chunk of response.body) {",
    "  process.stdout.write(decoder.decode(chunk, { stream: true }));",
    "}",
  ].join("\n");
  return { url, snippets: { curl, python, javascript }, shortened };
}

function shellQuote(text: string): string {
  return `'${text.replace(/'/g, `'\\''`)}'`;
}

function indent(text: string, spaces: number): string {
  const pad = " ".repeat(spaces);
  return text
    .split("\n")
    .map((line, i) => (i === 0 ? line : pad + line))
    .join("\n");
}
