import type { Connection } from "../api.ts";

export type Source = {
  id: string;
  title: string;
  heading: string;
  url: string;
  slug: string;
  anchor: string;
};
export type ChatMessage = { role: "user" | "assistant"; content: string };
export type AskEvent =
  | { type: "sources"; id: string; model: string; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done"; id: string }
  | { type: "error"; code: string };
export type AskState = {
  ready: boolean;
  code?: string;
  command?: string;
  model?: string;
  question: number;
  history: number;
};

/** getRandomValues also works on LAN HTTP, where randomUUID is unavailable. */
export function makeTabId(): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

export function docsEndpoint(connection: Connection, path: string): string {
  const url = new URL(connection.baseUrl);
  url.pathname =
    url.pathname.replace(/\/(?:v1)?\/?$/, "") + "/console/api/docs/" + path;
  url.search = "";
  url.hash = "";
  return url.href;
}
export async function docsRequest(
  connection: Connection,
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<Response> {
  const response = await fetch(docsEndpoint(connection, path), {
    method: body === undefined ? "GET" : "POST",
    signal,
    headers: {
      ...(connection.token
        ? { Authorization: `Bearer ${connection.token.trim()}` }
        : {}),
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(
      data.detail?.code ??
        (response.status === 401 || response.status === 403
          ? "auth_required"
          : "engine_error"),
    );
  }
  return response;
}

/** Incremental SSE framing supports split UTF-8 and multiple events per read. */
export async function readAskStream(
  response: Response,
  receive: (event: AskEvent) => void,
): Promise<void> {
  if (!response.body) throw new Error("stream_incomplete");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "",
    done = false;
  try {
    while (true) {
      const next = await reader.read();
      buffer = (
        buffer + decoder.decode(next.value, { stream: !next.done })
      ).replace(/\r\n/g, "\n");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const kind = /^event: (.+)$/m.exec(frame)?.[1];
        const data = frame
          .split("\n")
          .filter((l) => l.startsWith("data:"))
          .map((l) => l.slice(5).trimStart())
          .join("\n");
        if (!kind || !data) continue;
        const event = { ...JSON.parse(data), type: kind } as AskEvent;
        receive(event);
        if (event.type === "done") done = true;
        if (event.type === "error") throw new Error(event.code);
      }
      if (next.done) break;
    }
    if (!done) throw new Error("stream_incomplete");
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
}
