"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { authHeaders } from "./auth";

/**
 * Thin typed client over the backend (proxied by next.config.js). All paths are
 * absolute from the app origin: `/v1/...`, `/api/v1/...`, `/health`. Every
 * request carries the operator's Bearer token (when set) so authenticated
 * surfaces (model details, monitoring, load/unload) work — see lib/auth.
 */

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.error?.message ?? body?.detail ?? body?.message ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal) =>
    fetch(path, { headers: authHeaders(), signal }).then((r) => parse<T>(r)),

  delete: <T>(path: string, signal?: AbortSignal) =>
    fetch(path, { method: "DELETE", headers: authHeaders(), signal }).then((r) => parse<T>(r)),

  post: <T>(path: string, body?: unknown, signal?: AbortSignal) =>
    fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    }).then((r) => parse<T>(r)),

  patch: <T>(path: string, body: unknown, signal?: AbortSignal) =>
    fetch(path, {
      method: "PATCH",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(body),
      signal,
    }).then((r) => parse<T>(r)),

  postForm: <T>(path: string, form: FormData, signal?: AbortSignal) =>
    fetch(path, { method: "POST", headers: authHeaders(), body: form, signal }).then((r) =>
      parse<T>(r),
    ),

  /** POST returning a binary body (e.g. audio/wav). Accepts a JSON body or FormData. */
  postBlob: async (path: string, body: unknown, signal?: AbortSignal): Promise<Blob> => {
    const isForm = body instanceof FormData;
    const res = await fetch(path, {
      method: "POST",
      headers: isForm ? authHeaders() : { "Content-Type": "application/json", ...authHeaders() },
      body: isForm ? (body as FormData) : JSON.stringify(body),
      signal,
    });
    if (!res.ok) throw new ApiError(await res.text().catch(() => res.statusText), res.status);
    return res.blob();
  },

  /** Liveness probe used by the sidebar / settings. */
  health: async (signal?: AbortSignal): Promise<boolean> => {
    try {
      const r = await fetch("/health", { signal });
      return r.ok;
    } catch {
      return false;
    }
  },
};

/** A parsed SSE frame's data, with the SSE `event:` name (if any) attached as `_event`. */
export type SSEChunk = Record<string, unknown> & { _event?: string };

/**
 * POST a streaming (SSE) request and invoke `onChunk` for each parsed frame.
 * Handles both plain `data: {...}` streams (OpenAI/Yunshu) and `event: <name>`
 * + `data:` streams (Responses / Anthropic) — the event name is surfaced as
 * `_event` on the emitted object. Terminates on `data: [DONE]`, stream end, or
 * abort. `body` may be a JSON-serializable value or a FormData. The caller owns
 * the AbortController for cancellation.
 */
export async function streamSSE(
  path: string,
  body: unknown,
  onChunk: (data: SSEChunk) => void,
  signal?: AbortSignal,
): Promise<void> {
  const isForm = body instanceof FormData;
  const res = await fetch(path, {
    method: "POST",
    headers: isForm ? authHeaders() : { "Content-Type": "application/json", ...authHeaders() },
    body: isForm ? (body as FormData) : JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new ApiError(await res.text().catch(() => res.statusText), res.status);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r/g, "");
    // SSE frames are separated by a blank line.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      let event: string | undefined;
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
        // ignore id:, retry:, `:` comments / keep-alives
      }
      if (dataLines.length === 0) continue;
      const payload = dataLines.join("\n").trim();
      if (payload === "[DONE]") return;
      try {
        const obj = JSON.parse(payload) as Record<string, unknown>;
        onChunk(event ? { ...obj, _event: event } : obj);
      } catch {
        /* keep-alive / non-JSON frame */
      }
    }
  }
}

/**
 * Poll `fn` on an interval, exposing its latest result. Pauses when the tab is
 * hidden and cleans up on unmount. `fn` receives an AbortSignal.
 */
export function usePolling<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  intervalMs: number,
  deps: unknown[] = [],
): { data: T | null; error: Error | null; loading: boolean; refresh: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const tick = useCallback(async (signal: AbortSignal) => {
    try {
      const result = await fnRef.current(signal);
      if (!signal.aborted) {
        setData(result);
        setError(null);
      }
    } catch (e) {
      if (!signal.aborted) setError(e as Error);
    } finally {
      if (!signal.aborted) setLoading(false);
    }
  }, []);

  const [nonce, setNonce] = useState(0);
  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    tick(controller.signal);
    const id = setInterval(() => {
      if (document.visibilityState === "visible") tick(controller.signal);
    }, intervalMs);
    return () => {
      controller.abort();
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, nonce, ...deps]);

  return { data, error, loading, refresh };
}
