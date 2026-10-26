import { connectionScope, useScopedState } from "./scoped-state.ts";
import { useEffect, useRef, useState } from "react";
import { has, t, tr } from "./i18n/index.ts";
import { ApiError, type Connection } from "./api.ts";
import type { Offsets, Outcome, Row } from "./RequestTrace";
import { createSingleFlight, withTimeout } from "./single-flight.ts";
import { parseEnergy, parseLatency } from "./latency-api.ts";

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
  /** Recorded by the console process: the total duration, kept when the stage offsets are not. */
  total_ms?: number | null;
  ttft_ms?: number | null;
  prompt_tokens?: number | null;
  cached_tokens?: number | null;
  completion_tokens?: number | null;
  prefill_tps?: number | null;
  decode_tps?: number | null;
  cache?: { tier?: string | null; reload_ms?: number | null } | null;
  speculative?: Row["speculative"];
  cancelled?: boolean;
  latency?: unknown;
  energy?: unknown;
};

/** The finish reason as words ("length limit"), from the OpenAI and Anthropic vocabularies and the engine's own. */
export function finishLabel(reason: string): string {
  const key = reason.toLowerCase().replace(/[^a-z_]/g, "_");
  // i18n-keys: requests.finish.
  return has(`requests.finish.${key}`) ? tr(`requests.finish.${key}`) : reason;
}

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
    latency: "latency" in e ? parseLatency(e.latency) : undefined,
    energy: "energy" in e ? parseEnergy(e.energy) : undefined,
    source: "ring",
  };
}

/** A row read back from the console process's recorded log: metadata only, older than the ring may reach. */
export const historyToRow = (e: RecentEntry): Row => {
  const row = recentToRow(e);
  return {
    ...row,
    // The log keeps only the total of the stages: it still fills the Total column.
    offsets_ms:
      row.offsets_ms ??
      (typeof e.total_ms === "number" ? { done: e.total_ms } : row.offsets_ms),
    source: "history",
  };
};

export type Recent = {
  /** null until the first answer; false when the endpoint is missing (older server). */
  supported: boolean | null;
  rows: Row[];
  capacity: number;
  error: string | null;
};

const POLL_MS = 5000;
/** One answer may take this long; a wedged engine must not leave the poller waiting for ever. */
export const FETCH_TIMEOUT_MS = 10_000;

/** `requestJson` rejects query strings, so this one GET builds its own URL (same base, same bearer). */
export async function getRecent(
  connection: Connection,
  signal: AbortSignal,
  timeoutMs = FETCH_TIMEOUT_MS,
) {
  const url = new URL(connection.baseUrl.trim());
  const path = url.pathname.replace(/\/+$/, "");
  url.pathname = `${path.endsWith("/v1") ? path : `${path}/v1`}/yunshu/requests/recent`;
  url.search = "?limit=512";
  url.hash = "";
  const headers = new Headers({ Accept: "application/json" });
  if (connection.token.trim())
    headers.set("Authorization", `Bearer ${connection.token.trim()}`);
  const response = await fetch(url, {
    headers,
    signal: withTimeout(signal, timeoutMs),
  });
  if (!response.ok)
    throw new ApiError(`HTTP ${response.status}`, response.status);
  return (await response.json()) as { data?: RecentEntry[]; capacity?: number };
}

/** Pages of `GET /v1/yunshu/requests/history` (the console process's recorded log), newest first. */
async function getHistory(
  connection: Connection,
  signal: AbortSignal,
  pages = 2,
): Promise<RecentEntry[]> {
  const out: RecentEntry[] = [];
  let before: string | null = null;
  for (let i = 0; i < pages; i++) {
    const url = new URL(connection.baseUrl.trim());
    const path = url.pathname.replace(/\/+$/, "");
    url.pathname = `${path.endsWith("/v1") ? path : `${path}/v1`}/yunshu/requests/history`;
    url.search = `?limit=512${before ? `&before=${encodeURIComponent(before)}` : ""}`;
    url.hash = "";
    const headers = new Headers({ Accept: "application/json" });
    if (connection.token.trim())
      headers.set("Authorization", `Bearer ${connection.token.trim()}`);
    const response = await fetch(url, {
      headers,
      signal: withTimeout(signal, FETCH_TIMEOUT_MS),
    });
    if (!response.ok)
      throw new ApiError(`HTTP ${response.status}`, response.status);
    const body = (await response.json()) as {
      data?: RecentEntry[];
      next_cursor?: string | null;
    };
    if (Array.isArray(body.data)) out.push(...body.data);
    before = body.next_cursor ?? null;
    if (!before) break;
  }
  return out;
}

/**
 * Finished requests from the server ring. Refetches every 5 s while the tab is visible and
 * whenever `signal` (the latest finished id seen in status) changes. A 404 means an older
 * server without the endpoint: report `supported: false` and stop asking.
 */
export function useRecentRequests(
  connection: Connection,
  signal: string | null | undefined,
  /** The answers come through the console process, which keeps the recorded request log. */
  recorded = false,
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
  const historyRef = useRef<Row[]>([]);
  useEffect(() => {
    missing.current = false;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const schedule = () => {
      if (timer) clearTimeout(timer);
      if (!controller.signal.aborted && !missing.current)
        timer = setTimeout(() => void load(), POLL_MS);
    };
    // A kick while a fetch is in flight queues one more fetch instead of being dropped.
    const fetchOnce = createSingleFlight(async () => {
      try {
        const body = await getRecent(connection, controller.signal);
        if (controller.signal.aborted) return;
        const data = Array.isArray(body.data) ? body.data : [];
        const ring = data.filter((e) => e?.request_id).map(recentToRow);
        const ids = new Set(ring.map((r) => r.id));
        setState({
          supported: true,
          // the recorded log fills in what finished before the ring (or before this page opened)
          rows: [...ring, ...historyRef.current.filter((r) => !ids.has(r.id))],
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
          // The engine is away but the console's own record is not: the list stays what it was
          // (or, on a page opened during the outage, what the console recorded).
          ...(s.rows.length === 0 && historyRef.current.length > 0
            ? { supported: true, rows: [...historyRef.current] }
            : {}),
          error:
            e instanceof Error ? e.message : t("requests.recent.loadFailed"),
        }));
      }
    });
    const load = async () => {
      if (missing.current || controller.signal.aborted) return;
      if (document.visibilityState === "hidden") {
        schedule();
        return;
      }
      await fetchOnce();
      schedule();
    };
    kick.current = () => {
      if (timer) clearTimeout(timer);
      void load();
    };
    // Once per connection: what the console process recorded while nobody was looking.
    (recorded
      ? getHistory(connection, controller.signal)
      : Promise.resolve([])
    ).then(
      (entries) => {
        if (controller.signal.aborted) return;
        historyRef.current = entries
          .filter((e) => e?.request_id)
          .map(historyToRow);
        // `kick` re-reads the ring; with the engine away it fails and the recorded rows stand in.
        kick.current?.();
      },
      () => undefined,
    );
    void load();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [connection.baseUrl, connection.token, recorded]);
  useEffect(() => {
    if (signal) kick.current?.();
  }, [signal]);
  return state;
}
