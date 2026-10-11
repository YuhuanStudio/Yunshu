import { failureMessage, statusMessage } from "./errors.ts";
import { t } from "./i18n/index.ts";

export interface StreamConnection {
  baseUrl: string;
  token: string;
}

export interface CompletionMessage {
  role: string;
  content:
    | string
    | readonly (
        | { type: "text"; text: string }
        | { type: "image_url"; image_url: { url: string } }
      )[];
}

export interface CompletionBody {
  model: string;
  messages: CompletionMessage[];
  temperature: number;
  max_tokens: number;
  stream?: boolean;
  enable_thinking?: boolean;
  response_format?: { type: "json_object" };
  /** Chat dialect only: ask for the probability of every chosen token (and its runner-ups). */
  logprobs?: boolean;
  top_logprobs?: number;
  stream_options?: { include_usage: boolean };
}

export interface CompletionUsage {
  promptTokens?: number;
  completionTokens?: number;
  cachedTokens?: number;
  ttftMs?: number;
  /** x_yunshu.speculative: speculative decoding as the engine ran it. */
  spec?: {
    mode: string;
    rounds?: number;
    acceptanceRate?: number;
  };
}

/** One chosen token with its log-probability and the engine's runner-up candidates. */
export interface TokenLogprob {
  token: string;
  logprob: number;
  alternatives?: { token: string; logprob: number }[];
}

export interface CompletionDelta {
  content?: string;
  /** OpenAI `choices[0].logprobs.content`: the tokens this chunk added. */
  tokens?: TokenLogprob[];
  reasoning?: string;
  finishReason?: string;
  usage?: CompletionUsage;
}

const finiteNumber = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined;

/**
 * `choices[0].logprobs.content` of a chat chunk: every entry needs a string token and a finite
 * logprob, anything else is dropped (a malformed row never becomes a 0% token).
 */
export function parseLogprobs(logprobs: unknown): TokenLogprob[] {
  const content =
    logprobs && typeof logprobs === "object"
      ? (logprobs as Record<string, unknown>).content
      : undefined;
  if (!Array.isArray(content)) return [];
  const out: TokenLogprob[] = [];
  for (const entry of content) {
    if (!entry || typeof entry !== "object") continue;
    const e = entry as Record<string, unknown>;
    if (typeof e.token !== "string" || !finiteNumber(e.logprob)) continue;
    const tops = Array.isArray(e.top_logprobs)
      ? e.top_logprobs.flatMap((a): { token: string; logprob: number }[] => {
          const r = a as Record<string, unknown> | null;
          return r &&
            typeof r.token === "string" &&
            finiteNumber(r.logprob) !== undefined
            ? [{ token: r.token, logprob: r.logprob as number }]
            : [];
        })
      : [];
    out.push({
      token: e.token,
      logprob: e.logprob as number,
      ...(tops.length ? { alternatives: tops } : {}),
    });
  }
  return out;
}

/** Read OpenAI `usage` (plus optional x_yunshu extras) from a stream chunk. */
export function parseUsage(
  record: Record<string, unknown>,
): CompletionUsage | undefined {
  const usage =
    record.usage && typeof record.usage === "object"
      ? (record.usage as Record<string, unknown>)
      : undefined;
  const extra =
    record.x_yunshu && typeof record.x_yunshu === "object"
      ? (record.x_yunshu as Record<string, unknown>)
      : undefined;
  if (!usage && !extra) return undefined;
  const details =
    usage?.prompt_tokens_details &&
    typeof usage.prompt_tokens_details === "object"
      ? (usage.prompt_tokens_details as Record<string, unknown>)
      : undefined;
  const out: CompletionUsage = {};
  const promptTokens = finiteNumber(usage?.prompt_tokens);
  const completionTokens = finiteNumber(usage?.completion_tokens);
  const cachedTokens =
    finiteNumber(details?.cached_tokens) ?? finiteNumber(extra?.cached_tokens);
  const ttftMs = finiteNumber(extra?.ttft_ms);
  if (promptTokens !== undefined) out.promptTokens = promptTokens;
  if (completionTokens !== undefined) out.completionTokens = completionTokens;
  if (cachedTokens !== undefined) out.cachedTokens = cachedTokens;
  if (ttftMs !== undefined) out.ttftMs = ttftMs;
  const spec =
    extra?.speculative && typeof extra.speculative === "object"
      ? (extra.speculative as Record<string, unknown>)
      : undefined;
  if (spec && typeof spec.mode === "string") {
    out.spec = { mode: spec.mode };
    const rounds = finiteNumber(spec.rounds),
      rate = finiteNumber(spec.acceptance_rate);
    if (rounds !== undefined) out.spec.rounds = rounds;
    if (rate !== undefined) out.spec.acceptanceRate = rate;
  }
  return Object.keys(out).length ? out : undefined;
}

/** The three wire formats the engine serves. */
export type Dialect = "chat" | "responses" | "messages";

export const DIALECT_PATH: Record<Dialect, string> = {
  chat: "/chat/completions",
  responses: "/responses",
  messages: "/messages",
};

/** Normalize a server URL or API base URL to the endpoint of one dialect. */
export function endpointUrl(
  baseUrl: string,
  dialect: Dialect = "chat",
): string {
  let url: URL;
  try {
    url = new URL(baseUrl.trim());
  } catch {
    throw new Error("Invalid API base URL.");
  }

  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("API base URL must use http or https.");
  }
  if (url.username || url.password) {
    throw new Error("API base URL must not contain credentials.");
  }
  if (url.search || url.hash) {
    throw new Error(
      "API base URL must not contain a query string or fragment.",
    );
  }

  let path = url.pathname.replace(/\/+$/, "");
  for (const suffix of Object.values(DIALECT_PATH)) {
    if (path.endsWith(suffix)) {
      path = path.slice(0, -suffix.length);
      break;
    }
  }
  const apiPath = path.endsWith("/v1") ? path : `${path}/v1`;
  url.pathname = `${apiPath}${DIALECT_PATH[dialect]}`;
  return url.toString();
}

/** Normalize a server URL or API base URL to the OpenAI chat completions endpoint. */
export function chatCompletionsUrl(baseUrl: string): string {
  return endpointUrl(baseUrl, "chat");
}

const IMAGE_DATA_URL = /^data:([^;,]+);base64,(.*)$/s;

/** The JSON body one dialect expects for a chat-shaped request. */
export function buildPayload(
  dialect: Dialect,
  body: CompletionBody,
): Record<string, unknown> {
  if (dialect === "chat") return { ...body, stream: true };
  const system = body.messages
    .filter((m) => m.role === "system")
    .map((m) => (typeof m.content === "string" ? m.content : ""))
    .filter(Boolean)
    .join("\n\n");
  const turns = body.messages.filter((m) => m.role !== "system");
  if (dialect === "responses") {
    return {
      model: body.model,
      ...(system ? { instructions: system } : {}),
      input: turns.map((m) => ({
        role: m.role,
        content:
          typeof m.content === "string"
            ? m.content
            : m.content.map((part) =>
                part.type === "text"
                  ? {
                      type:
                        m.role === "assistant" ? "output_text" : "input_text",
                      text: part.text,
                    }
                  : { type: "input_image", image_url: part.image_url.url },
              ),
      })),
      temperature: body.temperature,
      max_output_tokens: body.max_tokens,
      stream: true,
      ...(body.enable_thinking !== undefined
        ? { enable_thinking: body.enable_thinking }
        : {}),
      ...(body.response_format
        ? { text: { format: { type: "json_object" } } }
        : {}),
    };
  }
  return {
    model: body.model,
    ...(system ? { system } : {}),
    messages: turns.map((m) => ({
      role: m.role,
      content:
        typeof m.content === "string"
          ? m.content
          : m.content.map((part) => {
              if (part.type === "text")
                return { type: "text", text: part.text };
              const match = IMAGE_DATA_URL.exec(part.image_url.url);
              return match
                ? {
                    type: "image",
                    source: {
                      type: "base64",
                      media_type: match[1],
                      data: match[2],
                    },
                  }
                : {
                    type: "image",
                    source: { type: "url", url: part.image_url.url },
                  };
            }),
    })),
    max_tokens: body.max_tokens,
    temperature: body.temperature,
    stream: true,
    // Messages has no JSON mode: response_format is not sent.
    ...(body.enable_thinking === undefined
      ? {}
      : body.enable_thinking
        ? {
            thinking: {
              type: "enabled",
              budget_tokens: Math.max(1, Math.floor(body.max_tokens / 2)),
            },
          }
        : { thinking: { type: "disabled" } }),
  };
}

const num = (v: unknown) => finiteNumber(v);
const obj = (v: unknown): Record<string, unknown> | undefined =>
  v && typeof v === "object" ? (v as Record<string, unknown>) : undefined;

function errorMessage(error: unknown, fallback: string): string {
  const e = obj(error);
  if (typeof e?.message === "string") return e.message;
  if (typeof error === "string") return error;
  return fallback;
}

/**
 * Read one Responses API event. Returns true on the terminal event.
 * Usage (and x_yunshu, which the engine nests inside usage) arrives with it.
 */
function handleResponsesEvent(
  type: string,
  record: Record<string, unknown>,
  onDelta: (delta: CompletionDelta) => void,
): boolean {
  if (type === "error" || type === "response.failed") {
    const response = obj(record.response);
    throw new Error(
      `OpenAI stream error: ${errorMessage(record.error ?? response?.error ?? record.message, JSON.stringify(record))}`,
    );
  }
  if (type === "response.output_text.delta" && typeof record.delta === "string")
    onDelta({ content: record.delta });
  else if (
    (type === "response.reasoning_summary_text.delta" ||
      type === "response.reasoning_text.delta") &&
    typeof record.delta === "string"
  )
    onDelta({ reasoning: record.delta });
  if (type !== "response.completed" && type !== "response.incomplete")
    return false;
  const response = obj(record.response) ?? {};
  const usage = obj(response.usage);
  if (usage) {
    const details = obj(usage.input_tokens_details);
    const parsed = parseUsage({
      usage: {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        prompt_tokens_details: details,
      },
      x_yunshu: usage.x_yunshu ?? response.x_yunshu,
    });
    if (parsed) onDelta({ usage: parsed });
  }
  const reason = obj(response.incomplete_details)?.reason;
  onDelta({
    finishReason:
      type === "response.incomplete"
        ? reason === "max_output_tokens"
          ? "length"
          : typeof reason === "string"
            ? reason
            : "incomplete"
        : "stop",
  });
  return true;
}

/** Read one Anthropic Messages stream event. Returns true on `message_stop`. */
function handleMessagesEvent(
  type: string,
  record: Record<string, unknown>,
  state: { input: number; cached: number },
  onDelta: (delta: CompletionDelta) => void,
): boolean {
  if (type === "error")
    throw new Error(
      `OpenAI stream error: ${errorMessage(record.error, JSON.stringify(record))}`,
    );
  if (type === "message_start") {
    const usage = obj(obj(record.message)?.usage);
    const cached = num(usage?.cache_read_input_tokens) ?? 0;
    // Anthropic's input_tokens excludes cache reads and writes; the prompt is the sum.
    state.input =
      (num(usage?.input_tokens) ?? 0) +
      cached +
      (num(usage?.cache_creation_input_tokens) ?? 0);
    state.cached = cached;
    return false;
  }
  if (type === "content_block_delta") {
    const delta = obj(record.delta);
    if (delta?.type === "text_delta" && typeof delta.text === "string")
      onDelta({ content: delta.text });
    else if (
      delta?.type === "thinking_delta" &&
      typeof delta.thinking === "string"
    )
      onDelta({ reasoning: delta.thinking });
    return false;
  }
  if (type === "message_delta") {
    const usage = obj(record.usage);
    const stop = obj(record.delta)?.stop_reason;
    const out = num(usage?.output_tokens);
    const parsed = parseUsage({
      usage: {
        prompt_tokens: state.input || undefined,
        completion_tokens: out,
        prompt_tokens_details: state.cached
          ? { cached_tokens: state.cached }
          : undefined,
      },
      x_yunshu: usage?.x_yunshu ?? record.x_yunshu,
    });
    if (parsed) onDelta({ usage: parsed });
    if (typeof stop === "string")
      onDelta({ finishReason: stop === "max_tokens" ? "length" : "stop" });
    return false;
  }
  return type === "message_stop";
}

function abortReason(signal: AbortSignal): Error {
  if (signal.reason instanceof Error) return signal.reason;
  return new DOMException("The operation was aborted.", "AbortError");
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw abortReason(signal);
}

/** HTTP failure with a localised `message`; the backend's own text stays in `detail`. */
export class StreamHttpError extends Error {
  readonly status: number;
  readonly detail: string;
  constructor(status: number, statusText: string, detail: string) {
    super(statusMessage(status));
    this.name = "StreamHttpError";
    this.status = status;
    this.detail = [
      `HTTP ${status}${statusText ? ` ${statusText}` : ""}`,
      detail,
    ]
      .filter(Boolean)
      .join("\n");
  }
}

/** Short localised text plus optional raw details for any failure of a stream. */
export function describeStreamError(error: unknown): {
  message: string;
  detail?: string;
} {
  if (error instanceof StreamHttpError)
    return { message: error.message, detail: error.detail };
  if (error instanceof TypeError)
    return { message: failureMessage("network"), detail: error.message };
  if (error instanceof DOMException && error.name === "TimeoutError")
    return { message: failureMessage("timeout"), detail: error.message };
  if (error instanceof Error)
    return { message: t("errors.stream.interrupted"), detail: error.message };
  return { message: t("errors.stream.failed") };
}

async function httpError(response: Response): Promise<Error> {
  let detail = "";
  try {
    const text = await response.text();
    if (text) {
      try {
        const payload: unknown = JSON.parse(text);
        if (payload && typeof payload === "object") {
          const record = payload as Record<string, unknown>;
          const error = record.error;
          if (typeof error === "string") detail = error;
          else if (
            error &&
            typeof error === "object" &&
            typeof (error as Record<string, unknown>).message === "string"
          ) {
            detail = (error as Record<string, unknown>).message as string;
          } else if (typeof record.detail === "string") detail = record.detail;
          else if (typeof record.message === "string") detail = record.message;
        }
      } catch {
        detail = text.trim();
      }
    }
  } catch {
    // The status and statusText still provide a useful error if the body is unreadable.
  }

  return new StreamHttpError(response.status, response.statusText, detail);
}

/**
 * Stream OpenAI-compatible chat completions and forward content/reasoning deltas.
 * The supplied AbortSignal owns both the request and the response reader.
 */
export async function streamCompletion(
  connection: StreamConnection,
  body: CompletionBody,
  onDelta: (delta: CompletionDelta) => void,
  signal: AbortSignal,
  dialect: Dialect = "chat",
): Promise<void> {
  throwIfAborted(signal);

  const headers = new Headers({
    Accept: "text/event-stream",
    "Content-Type": "application/json",
  });
  if (connection.token.trim())
    headers.set("Authorization", `Bearer ${connection.token}`);
  const requestId = globalThis.crypto?.randomUUID?.();
  if (requestId) headers.set("X-Request-ID", requestId);

  const response = await fetch(endpointUrl(connection.baseUrl, dialect), {
    method: "POST",
    headers,
    body: JSON.stringify(buildPayload(dialect, body)),
    signal,
  });
  throwIfAborted(signal);
  if (!response.ok) throw await httpError(response);
  if (!response.body)
    throw new Error("OpenAI API response did not include a stream body.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let dataLines: string[] = [];
  let eventType = "";
  let receivedDone = false;
  const messagesState = { input: 0, cached: 0 };

  const dispatchEvent = () => {
    const data = dataLines.join("\n");
    dataLines = [];
    const type = eventType;
    eventType = "";
    if (!data) return;
    if (data.trim() === "[DONE]") {
      receivedDone = true;
      return;
    }

    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      const message =
        type === "error" ? data : `Invalid JSON in OpenAI stream: ${data}`;
      throw new Error(message);
    }

    if (type === "error") {
      const record =
        payload && typeof payload === "object"
          ? (payload as Record<string, unknown>)
          : undefined;
      const error = record?.error;
      const message =
        error &&
        typeof error === "object" &&
        typeof (error as Record<string, unknown>).message === "string"
          ? ((error as Record<string, unknown>).message as string)
          : typeof error === "string"
            ? error
            : typeof record?.message === "string"
              ? record.message
              : data;
      throw new Error(`OpenAI stream error: ${message}`);
    }

    if (!payload || typeof payload !== "object") return;
    const record = payload as Record<string, unknown>;
    if (dialect !== "chat") {
      const kind = typeof record.type === "string" ? record.type : type;
      const finished =
        dialect === "responses"
          ? handleResponsesEvent(kind, record, onDelta)
          : handleMessagesEvent(kind, record, messagesState, onDelta);
      if (finished) receivedDone = true;
      return;
    }
    if (record.error != null) {
      const error = record.error;
      const message =
        error &&
        typeof error === "object" &&
        typeof (error as Record<string, unknown>).message === "string"
          ? ((error as Record<string, unknown>).message as string)
          : typeof error === "string"
            ? error
            : JSON.stringify(error);
      throw new Error(`OpenAI stream error: ${message}`);
    }

    const usage = parseUsage(record);
    if (usage) onDelta({ usage });

    const choices = record.choices;
    if (
      !Array.isArray(choices) ||
      !choices.length ||
      !choices[0] ||
      typeof choices[0] !== "object"
    )
      return;
    const choice = choices[0] as Record<string, unknown>;
    if (typeof choice.finish_reason === "string")
      onDelta({ finishReason: choice.finish_reason });
    const tokens = parseLogprobs(choice.logprobs);
    if (tokens.length) onDelta({ tokens });
    const delta = choice.delta;
    if (!delta || typeof delta !== "object") return;
    const deltaRecord = delta as Record<string, unknown>;
    const content =
      typeof deltaRecord.content === "string" ? deltaRecord.content : undefined;
    const reasoningValue =
      deltaRecord.reasoning_content ?? deltaRecord.reasoning;
    const reasoning =
      typeof reasoningValue === "string" ? reasoningValue : undefined;
    if (content !== undefined || reasoning !== undefined) {
      const chunk: CompletionDelta = {};
      if (content !== undefined) chunk.content = content;
      if (reasoning !== undefined) chunk.reasoning = reasoning;
      onDelta(chunk);
    }
  };

  const processLine = (line: string) => {
    if (receivedDone) return;
    if (line === "") {
      dispatchEvent();
      return;
    }
    if (line.startsWith(":")) return;

    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") dataLines.push(value);
    else if (field === "event") eventType = value;
  };

  const cancelOnAbort = () => {
    void reader.cancel(signal.reason).catch(() => {});
  };
  signal.addEventListener("abort", cancelOnAbort, { once: true });
  try {
    while (!receivedDone) {
      throwIfAborted(signal);
      const { value, done } = await reader.read();
      throwIfAborted(signal);
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      let newline = buffer.indexOf("\n");
      while (newline !== -1) {
        let line = buffer.slice(0, newline);
        if (line.endsWith("\r")) line = line.slice(0, -1);
        processLine(line);
        buffer = buffer.slice(newline + 1);
        if (receivedDone) break;
        newline = buffer.indexOf("\n");
      }
    }

    buffer += decoder.decode();
    if (!receivedDone && buffer.length) {
      processLine(buffer.endsWith("\r") ? buffer.slice(0, -1) : buffer);
    }
    if (!receivedDone) dispatchEvent();
    throwIfAborted(signal);
    if (!receivedDone)
      throw new Error(
        dialect === "chat"
          ? "OpenAI stream ended before the [DONE] event."
          : dialect === "responses"
            ? "OpenAI stream ended before the response.completed event."
            : "OpenAI stream ended before the message_stop event.",
      );
  } finally {
    signal.removeEventListener("abort", cancelOnAbort);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
