import {
  parseSchemaText,
  validateSchema,
  type SchemaIssue,
} from "./json-schema-lite.ts";

export interface ToolDef {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export type DefsCheck =
  { ok: true; tools: ToolDef[] } | { ok: false; errors: string[] };

const NAME = /^[a-zA-Z0-9_-]{1,64}$/;

/** Check the tool definitions text before anything is sent. Errors are codes plus the offending name or position. */
export function checkToolDefs(text: string): DefsCheck {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return {
      ok: false,
      errors: [`json:${e instanceof Error ? e.message : "parse"}`],
    };
  }
  if (!Array.isArray(raw) || raw.length === 0)
    return { ok: false, errors: ["array"] };
  const errors: string[] = [];
  const seen = new Set<string>();
  raw.forEach((t, i) => {
    const f = (t as { function?: unknown } | null)?.function as
      { name?: unknown; parameters?: unknown } | undefined;
    if (
      (t as { type?: unknown } | null)?.type !== "function" ||
      !f ||
      typeof f !== "object"
    ) {
      errors.push(`shape:${i + 1}`);
      return;
    }
    if (typeof f.name !== "string" || !NAME.test(f.name))
      errors.push(`name:${i + 1}`);
    else if (seen.has(f.name)) errors.push(`duplicate:${f.name}`);
    else seen.add(f.name);
    if (f.parameters !== undefined) {
      const p = f.parameters as { type?: unknown } | null;
      if (
        !p ||
        typeof p !== "object" ||
        Array.isArray(p) ||
        p.type !== "object"
      )
        errors.push(
          `parameters:${typeof f.name === "string" ? f.name : i + 1}`,
        );
    }
  });
  return errors.length
    ? { ok: false, errors }
    : { ok: true, tools: raw as ToolDef[] };
}

export interface ToolCall {
  id: string;
  name: string;
  /** The arguments exactly as the engine sent them. */
  rawArguments: string;
  /** Parsed arguments, or null when they are not valid JSON. */
  args: unknown;
  parseError: string | null;
  /** Violations against the tool's own parameters schema; empty when valid. */
  issues: SchemaIssue[];
  /** False when the engine called a tool that is not in the definitions. */
  known: boolean;
}

export interface Round {
  content: string;
  toolCalls: ToolCall[];
  finishReason: string | null;
  usage: { prompt: number | null; completion: number | null };
  /** The assistant message to send back verbatim with the tool results. */
  assistant: Record<string, unknown>;
}

const str = (v: unknown) => (typeof v === "string" ? v : "");
const num = (v: unknown) =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

/** Read one non-streaming chat completion into a round; null when it has no choice at all. */
export function readRound(
  body: unknown,
  tools: readonly ToolDef[],
): Round | null {
  const choice = (body as { choices?: unknown[] } | null)?.choices?.[0] as
    { message?: Record<string, unknown>; finish_reason?: unknown } | undefined;
  if (!choice?.message) return null;
  const message = choice.message;
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  const toolCalls = calls.map((c, i): ToolCall => {
    const call = c as {
      id?: unknown;
      function?: { name?: unknown; arguments?: unknown };
    };
    const name = str(call.function?.name);
    const raw =
      typeof call.function?.arguments === "string"
        ? call.function.arguments
        : JSON.stringify(call.function?.arguments ?? {});
    let args: unknown = null;
    let parseError: string | null = null;
    try {
      args = raw.trim() ? JSON.parse(raw) : {};
    } catch (e) {
      parseError = e instanceof Error ? e.message : "parse";
    }
    const def = tools.find((t) => t.function.name === name);
    const issues =
      def && parseError == null && def.function.parameters
        ? validateSchema(def.function.parameters, args)
        : [];
    return {
      id: str(call.id) || `call_${i + 1}`,
      name,
      rawArguments: raw,
      args,
      parseError,
      issues,
      known: !!def,
    };
  });
  const usage = (body as { usage?: Record<string, unknown> }).usage ?? {};
  return {
    content: str(message.content),
    toolCalls,
    finishReason:
      typeof choice.finish_reason === "string" ? choice.finish_reason : null,
    usage: {
      prompt: num(usage.prompt_tokens),
      completion: num(usage.completion_tokens),
    },
    assistant: {
      role: "assistant",
      content: message.content ?? null,
      ...(toolCalls.length
        ? {
            tool_calls: toolCalls.map((c) => ({
              id: c.id,
              type: "function",
              function: { name: c.name, arguments: c.rawArguments },
            })),
          }
        : {}),
    },
  };
}

export type ChatMessage = Record<string, unknown>;

/** The messages for the next round: the assistant's tool call message, then one tool message per call. */
export function nextMessages(
  messages: readonly ChatMessage[],
  round: Round,
  results: Readonly<Record<string, string>>,
): ChatMessage[] {
  return [
    ...messages,
    round.assistant,
    ...round.toolCalls.map((c) => ({
      role: "tool",
      tool_call_id: c.id,
      content: results[c.id] ?? "",
    })),
  ];
}

/** Structured output check: the reply must be JSON, then must satisfy the schema. */
export function checkStructured(
  content: string,
  schemaText: string,
):
  | { state: "empty" }
  | { state: "notJson"; message: string }
  | { state: "bad"; issues: SchemaIssue[] }
  | { state: "ok" } {
  if (!content.trim()) return { state: "empty" };
  let value: unknown;
  try {
    value = JSON.parse(content);
  } catch (e) {
    return {
      state: "notJson",
      message: e instanceof Error ? e.message : "parse",
    };
  }
  const parsed = parseSchemaText(schemaText);
  if ("error" in parsed) return { state: "ok" };
  const issues = validateSchema(parsed.schema, value);
  return issues.length ? { state: "bad", issues } : { state: "ok" };
}
