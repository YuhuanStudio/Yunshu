import { failureMessage, statusMessage } from "./errors.ts";
import { ApiError, type Connection, type RequestOptions } from "./api.ts";

export interface ServerRequestOptions extends RequestOptions {
  /** Permit an empty success body for Ollama-compatible management calls. */
  allowEmpty?: boolean;
}

const DEFAULT_TIMEOUT_MS = 10_000;
const PULL_TIMEOUT_MS = 30 * 60 * 1_000;

function serverRoot(connection: Connection): URL {
  let url: URL;
  try {
    url = new URL(connection.baseUrl.trim());
  } catch {
    throw new ApiError("Enter a valid Yunshu server address.");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:")
    throw new ApiError("The Yunshu address must use HTTP or HTTPS.");
  if (url.username || url.password)
    throw new ApiError("Do not put credentials in the Yunshu address.");
  if (url.search || url.hash)
    throw new ApiError(
      "The Yunshu address cannot include a query or fragment.",
    );
  url.pathname = url.pathname.replace(/\/+$/, "").replace(/\/v1$/i, "") + "/";
  url.search = "";
  url.hash = "";
  return url;
}

function serverUrl(connection: Connection, path: string): URL {
  if (
    !path.startsWith("/") ||
    path.startsWith("//") ||
    path.includes("?") ||
    path.includes("#") ||
    path.includes("\\")
  )
    throw new ApiError("Invalid Yunshu server path.");
  if (path.split("/").some((part) => part === "." || part === ".."))
    throw new ApiError("Invalid Yunshu server path.");
  return new URL(path.slice(1), serverRoot(connection));
}

function detailFrom(payload: unknown): unknown {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload))
    return undefined;
  const record = payload as Record<string, unknown>;
  return record.detail ?? record.message ?? record.error;
}

function publicError(
  status: number,
  detail: unknown,
  statusText: string,
): string {
  void detail;
  void statusText;
  return statusMessage(status);
}

/** Request a server-root API path (`/api/*` or `/debug/*`) with bearer auth. */
export async function requestServerJson<T = unknown>(
  connection: Connection,
  path: string,
  options: ServerRequestOptions = {},
): Promise<T | undefined> {
  const url = serverUrl(connection, path);
  const controller = new AbortController();
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  let didTimeout = false;
  const timeout = globalThis.setTimeout(() => {
    didTimeout = true;
    controller.abort(new DOMException("Request timed out", "TimeoutError"));
  }, timeoutMs);
  const forwardAbort = () => controller.abort(options.signal?.reason);
  if (options.signal?.aborted) forwardAbort();
  else options.signal?.addEventListener("abort", forwardAbort, { once: true });

  const headers = new Headers({ Accept: "application/json" });
  if (connection.token.trim())
    headers.set("Authorization", `Bearer ${connection.token.trim()}`);
  let body: string | undefined;
  if (options.body !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.body);
  }

  try {
    const response = await fetch(url, {
      method: options.method ?? "GET",
      headers,
      body,
      signal: controller.signal,
    });
    const text = await response.text();
    let payload: unknown;
    if (text.trim()) {
      try {
        payload = JSON.parse(text) as unknown;
      } catch {
        if (response.ok)
          throw new ApiError(failureMessage("json"), response.status);
      }
    } else if (response.ok && !options.allowEmpty) {
      throw new ApiError(failureMessage("empty"), response.status);
    }
    if (!response.ok) {
      const detail = detailFrom(payload);
      throw new ApiError(
        publicError(response.status, detail, response.statusText),
        response.status,
        detail,
      );
    }
    return payload as T | undefined;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (options.signal?.aborted) throw options.signal.reason ?? error;
    if (didTimeout || controller.signal.aborted)
      throw new ApiError(failureMessage("timeout"), undefined, undefined, {
        cause: error,
      });
    throw new ApiError(failureMessage("network"), undefined, undefined, {
      cause: error,
    });
  } finally {
    globalThis.clearTimeout(timeout);
    options.signal?.removeEventListener("abort", forwardAbort);
  }
}

export async function pullModel(
  connection: Connection,
  model: string,
  signal?: AbortSignal,
): Promise<void> {
  await requestServerJson(connection, "/api/pull", {
    method: "POST",
    body: { model, stream: false },
    signal,
    timeoutMs: PULL_TIMEOUT_MS,
    allowEmpty: true,
  });
}

export async function copyModel(
  connection: Connection,
  source: string,
  destination: string,
  signal?: AbortSignal,
): Promise<void> {
  await requestServerJson(connection, "/api/copy", {
    method: "POST",
    body: { source, destination },
    signal,
    allowEmpty: true,
  });
}

export async function deleteModel(
  connection: Connection,
  model: string,
  signal?: AbortSignal,
): Promise<void> {
  await requestServerJson(connection, "/api/delete", {
    method: "DELETE",
    body: { model },
    signal,
    allowEmpty: true,
  });
}
