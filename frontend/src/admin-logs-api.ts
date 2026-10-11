import { ApiError, type Connection } from "./api.ts";

/** One record of the server's in-memory log ring (already redacted by the server). */
export type LogRecord = {
  id: number;
  /** Unix seconds. */
  t: number;
  level: string;
  logger: string;
  msg: string;
};

export type LogPage = {
  records: LogRecord[];
  next_id: number;
  dropped: number;
  capacity: number;
  server_time: number;
};

export type LogQuery = {
  level?: string;
  since?: number;
  since_id?: number;
  q?: string;
  limit?: number;
};

/** `/v1/yunshu/<path>?query` on the connection's origin, with the bearer header when a token is set. */
function endpoint(
  connection: Connection,
  path: string,
  query: Record<string, string | number | undefined> = {},
) {
  const url = new URL(connection.baseUrl.trim());
  const base = url.pathname.replace(/\/+$/, "");
  url.pathname = `${base.endsWith("/v1") ? base : `${base}/v1`}${path}`;
  url.hash = "";
  url.search = "";
  for (const [k, v] of Object.entries(query))
    if (v !== undefined && v !== "") url.searchParams.set(k, String(v));
  const headers = new Headers();
  if (connection.token.trim())
    headers.set("Authorization", `Bearer ${connection.token.trim()}`);
  return { url, headers };
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** Accept only well-formed records; a malformed entry is dropped, never shown as zeros. */
export function parseRecord(value: unknown): LogRecord | null {
  if (!isRecord(value)) return null;
  const { id, t, level, logger, msg } = value;
  if (
    typeof id !== "number" ||
    typeof t !== "number" ||
    typeof msg !== "string"
  )
    return null;
  return {
    id,
    t,
    level: typeof level === "string" ? level.toUpperCase() : "INFO",
    logger: typeof logger === "string" ? logger : "",
    msg,
  };
}

export async function fetchLogs(
  connection: Connection,
  query: LogQuery,
  signal?: AbortSignal,
): Promise<LogPage> {
  const { url, headers } = endpoint(connection, "/yunshu/logs", { ...query });
  headers.set("Accept", "application/json");
  const response = await fetch(url, { headers, signal });
  if (!response.ok)
    throw new ApiError(`HTTP ${response.status}`, response.status);
  const body: unknown = await response.json();
  if (!isRecord(body) || !Array.isArray(body.records))
    throw new ApiError("bad shape", response.status);
  const num = (v: unknown, d: number) => (typeof v === "number" ? v : d);
  return {
    records: body.records.flatMap((r) => parseRecord(r) ?? []),
    next_id: num(body.next_id, 0),
    dropped: num(body.dropped, 0),
    capacity: num(body.capacity, 2000),
    server_time: num(body.server_time, Date.now() / 1000),
  };
}

/** Feed text chunks in; get complete SSE `data:` payloads out. Comments (keepalives) are ignored. */
export function createSseParser(onData: (data: string) => void) {
  let buffer = "";
  return (chunk: string) => {
    buffer += chunk.replace(/\r\n?/g, "\n");
    let at: number;
    while ((at = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, at);
      buffer = buffer.slice(at + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (data) onData(data);
    }
  };
}

/**
 * Live tail over `GET /v1/yunshu/logs/stream`. EventSource cannot send the bearer header, so
 * this reads the response body itself. Resolves when the stream ends or is aborted; rejects
 * with an ApiError when the server refuses (404 older engine, 401/403 missing admin).
 */
export async function streamLogs(
  connection: Connection,
  query: Pick<LogQuery, "level" | "q" | "since_id">,
  onRecord: (record: LogRecord) => void,
  signal: AbortSignal,
  onOpen?: () => void,
): Promise<void> {
  const { url, headers } = endpoint(connection, "/yunshu/logs/stream", {
    ...query,
  });
  headers.set("Accept", "text/event-stream");
  const response = await fetch(url, { headers, signal });
  if (!response.ok || !response.body)
    throw new ApiError(`HTTP ${response.status}`, response.status);
  onOpen?.();
  const feed = createSseParser((data) => {
    try {
      const record = parseRecord(JSON.parse(data));
      if (record) onRecord(record);
    } catch {
      /* a broken event is skipped; the next one is independent */
    }
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      feed(decoder.decode(value, { stream: true }));
    }
  } finally {
    reader.cancel().catch(() => undefined);
  }
}

/** Save text as a file through a temporary object URL. */
export function saveBlob(name: string, blob: Blob) {
  const href = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = href;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
}

/** `GET /v1/yunshu/bundle` (admin) saved with the server's file name. */
export async function downloadBundle(
  connection: Connection,
  signal?: AbortSignal,
): Promise<string> {
  const { url, headers } = endpoint(connection, "/yunshu/bundle");
  const response = await fetch(url, { headers, signal });
  if (!response.ok)
    throw new ApiError(`HTTP ${response.status}`, response.status);
  const blob = await response.blob();
  const named = /filename="?([^";]+)"?/.exec(
    response.headers.get("content-disposition") ?? "",
  )?.[1];
  const name = named ?? `yunshu-bundle-${Date.now()}.json`;
  saveBlob(name, blob);
  return name;
}

/** Plain-text line of one record, as copied and downloaded. */
export function formatLine(r: LogRecord): string {
  const d = new Date(r.t * 1000);
  const p = (n: number, w = 2) => String(n).padStart(w, "0");
  const stamp = `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)}`;
  return `${stamp} ${r.level.padEnd(7)} ${r.logger} ${r.msg}`;
}
