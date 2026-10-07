export interface StreamConnection {
  baseUrl: string;
  token: string;
}

export interface CompletionMessage {
  role: string;
  content: string;
}

export interface CompletionBody {
  model: string;
  messages: CompletionMessage[];
  temperature: number;
  max_tokens: number;
  stream?: boolean;
}

export interface CompletionDelta {
  content?: string;
  reasoning?: string;
}

/** Normalize a server URL or API base URL to the OpenAI chat completions endpoint. */
export function chatCompletionsUrl(baseUrl: string): string {
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

  const path = url.pathname.replace(/\/+$/, "");
  if (path.endsWith("/chat/completions")) {
    url.pathname = path;
  } else {
    const apiPath = path.endsWith("/v1") ? path : `${path}/v1`;
    url.pathname = `${apiPath}/chat/completions`;
  }
  return url.toString();
}

function abortReason(signal: AbortSignal): Error {
  if (signal.reason instanceof Error) return signal.reason;
  return new DOMException("The operation was aborted.", "AbortError");
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw abortReason(signal);
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

  const status = `${response.status}${response.statusText ? ` ${response.statusText}` : ""}`;
  return new Error(
    `OpenAI API request failed (${status})${detail ? `: ${detail}` : "."}`,
  );
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

  const response = await fetch(chatCompletionsUrl(connection.baseUrl), {
    method: "POST",
    headers,
    body: JSON.stringify({ ...body, stream: true }),
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

    const choices = record.choices;
    if (
      !Array.isArray(choices) ||
      !choices.length ||
      !choices[0] ||
      typeof choices[0] !== "object"
    )
      return;
    const delta = (choices[0] as Record<string, unknown>).delta;
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
      throw new Error("OpenAI stream ended before the [DONE] event.");
  } finally {
    signal.removeEventListener("abort", cancelOnAbort);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
