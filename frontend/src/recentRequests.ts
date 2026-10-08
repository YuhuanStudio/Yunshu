import { connectionScope, useScopedState } from "./scoped-state.ts";
import { useEffect, useRef, useState } from "react";
import { t } from "./i18n/index.ts";
import { ApiError, type Connection } from "./api";
import type { Offsets, Outcome, Row } from "./RequestTrace";

/** One entry of `GET /v1/yunshu/requests/recent` (numbers and enums only, never prompt text). */
export type RecentEntry = {
  request_id: string;
  t?: number;
  model?: string | null;
  path?: string | null;
  status?: number | string | null;
  finish_reason?: string | null;
  stream?: boolean | null;
  t0_wall?: number | null;
  offsets_ms?: Offsets | null;
  queue_wait_ms?: number | null;
  ttft_ms?: number | null;
  prompt_tokens?: number | null;
  cached_tokens?: number | null;
  completion_tokens?: number | null;
  prefill_tps?: number | null;
  decode_tps?: number | null;
  cache?: { tier?: string | null; reload_ms?: number | null } | null;
  speculative?: Row["speculative"];
  cancelled?: boolean;
};

export const outcomeLabel = (o: Outcome): string =>
  o === "completed"
    ? t("requests.outcome.completed")
    : o === "cancelled"
      ? t("requests.outcome.cancelled")
      : t("requests.outcome.error");

/** `status` is the HTTP status the server answered with; a cancel or an error never counts as completed. */
export function outcomeOf(
  e: Pick<RecentEntry, "status" | "cancelled">,
): Outcome {
  if (e.cancelled) return "cancelled";
  const code = typeof e.status === "string" ? Number(e.status) : e.status;
  if (typeof e.status === "string" && !Number.isFinite(code)) {
    const word = e.status.toLowerCase();
    if (word.startsWith("cancel")) return "cancelled";
    return word === "completed" || word === "ok" ? "completed" : "error";
  }
  return code != null && code >= 400 ? "error" : "completed";
}

export function recentToRow(e: RecentEntry): Row {
  return {
    id: e.request_id,
    phase: "complete",
    model: e.model ?? undefined,
    prompt_tokens: e.prompt_tokens ?? undefined,
    cached_tokens: e.cached_tokens ?? undefined,
    completion_tokens: e.completion_tokens ?? undefined,
    ttft_ms: e.ttft_ms,
    decode_tps: e.decode_tps,
    prefill_tps: e.prefill_tps,
    speculative: e.speculative,
    t: e.t,
    path: e.path ?? undefined,
    outcome: outcomeOf(e),
    status_code: typeof e.status === "number" ? e.status : null,
    finish_reason: e.finish_reason,
    offsets_ms: e.offsets_ms,
    cache: e.cache,
    queue_wait_ms: e.queue_wait_ms,
    stream: e.stream,
    t0_wall: e.t0_wall,
    source: "ring",
  };
}

export type Recent = {
  /** null until the first answer; false when the endpoint is missing (older server). */
  supported: boolean | null;
  rows: Row[];
  capacity: number;
  error: string | null;
};

const POLL_MS = 5000;

/** `requestJson` rejects query strings, so this one GET builds its own URL (same base, same bearer). */
async function getRecent(connection: Connection, signal: AbortSignal) {
  const url = new URL(connection.baseUrl.trim());
  const path = url.pathname.replace(/\/+$/, "");
  url.pathname = `${path.endsWith("/v1") ? path : `${path}/v1`}/yunshu/requests/recent`;
  url.search = "?limit=512";
  url.hash = "";
  const headers = new Headers({ Accept: "application/json" });
  if (connection.token.trim())
    headers.set("Authorization", `Bearer ${connection.token.trim()}`);
  const response = await fetch(url, { headers, signal });
  if (!response.ok)
    throw new ApiError(`HTTP ${response.status}`, response.status);
  return (await response.json()) as { data?: RecentEntry[]; capacity?: number };
}

/**
 * Finished requests from the server ring. Refetches every 5 s while the tab is visible and
 * whenever `signal` (the latest finished id seen in status) changes. A 404 means an older
 * server without the endpoint: report `supported: false` and stop asking.
 */
export function useRecentRequests(
  connection: Connection,
  signal: string | null | undefined,
): Recent {
  const [state, setState] = useScopedState<Recent>(
    connectionScope(connection),
    {
      supported: null,
      rows: [],
      capacity: 512,
      error: null,
    },
  );
  const missing = useRef(false);
  const kick = useRef<(() => void) | null>(null);
  useEffect(() => {
    missing.current = false;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined,
      flying = false;
    const load = async () => {
      if (missing.current || flying || controller.signal.aborted) return;
      if (document.visibilityState === "hidden") {
        timer = setTimeout(() => void load(), POLL_MS);
        return;
      }
      flying = true;
      try {
        const body = await getRecent(connection, controller.signal);
        if (controller.signal.aborted) return;
        const data = Array.isArray(body.data) ? body.data : [];
        setState({
          supported: true,
          rows: data.filter((e) => e?.request_id).map(recentToRow),
          capacity: body.capacity ?? 512,
          error: null,
        });
      } catch (e) {
        if (controller.signal.aborted) return;
        if (e instanceof ApiError && (e.status === 404 || e.status === 405)) {
          missing.current = true;
          setState((s) => ({ ...s, supported: false, rows: [], error: null }));
          return;
        }
        setState((s) => ({
          ...s,
          error:
            e instanceof Error ? e.message : t("requests.recent.loadFailed"),
        }));
      } finally {
        flying = false;
        if (!controller.signal.aborted && !missing.current) {
          timer = setTimeout(() => void load(), POLL_MS);
        }
      }
    };
    kick.current = () => {
      if (timer) clearTimeout(timer);
      void load();
    };
    void load();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [connection.baseUrl, connection.token]);
  useEffect(() => {
    if (signal) kick.current?.();
  }, [signal]);
  return state;
}
