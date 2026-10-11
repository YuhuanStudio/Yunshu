import { requestJson, type Connection } from "./api.ts";
import { binaryGb } from "./byte-format.ts";
import type { SeriesPoint } from "./series.ts";

export interface ServerHistory {
  points: SeriesPoint[];
  intervalS: number | null;
  /** Spans (epoch ms) with no samples at all: the engine process was not running. */
  gaps: [number, number][];
}

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

/**
 * Parse `GET /v1/yunshu/history` (columnar rows, epoch seconds). Anything that
 * is not the documented shape yields null, so an older or odd server simply
 * means "no engine-side history" and the charts start from live polls.
 */
export function parseServerHistory(
  payload: unknown,
  /** True when the engine reports binary GB (it also sends `*_bytes` in /status); older engines are decimal. */
  binary = false,
): ServerHistory | null {
  const gbOf = (v: unknown): number | null => {
    const n = num(v);
    return n == null ? null : binary ? n : binaryGb(n);
  };
  if (!payload || typeof payload !== "object") return null;
  const body = payload as Record<string, unknown>;
  if (body.enabled === false) return null;
  const series = body.series;
  if (!series || typeof series !== "object") return null;
  const cols = series as Record<string, unknown>;
  const t = cols.t;
  if (!Array.isArray(t) || t.length === 0) return null;
  const col = (name: string): unknown[] =>
    Array.isArray(cols[name]) && (cols[name] as unknown[]).length === t.length
      ? (cols[name] as unknown[])
      : [];
  const decode = col("decode_tps"),
    prefill = col("prefill_tps"),
    active = col("requests_active"),
    queued = col("queued"),
    memActive = col("active_gb"),
    memCache = col("cache_gb");
  const points: SeriesPoint[] = [];
  let previous = -Infinity;
  for (let i = 0; i < t.length; i++) {
    const seconds = num(t[i]);
    if (seconds == null) continue;
    const at = seconds * 1000;
    if (at <= previous) continue;
    previous = at;
    points.push({
      at,
      // a sample that has request counts was taken while the engine answered: no rate then is 0
      decode: num(decode[i]) ?? (num(active[i]) != null ? 0 : null),
      prefill: num(prefill[i]) ?? (num(active[i]) != null ? 0 : null),
      active: num(active[i]),
      queued: num(queued[i]),
      prefillRequests: null,
      decodeRequests: null,
      memActive: gbOf(memActive[i]),
      memCache: gbOf(memCache[i]),
      backfilled: true,
    });
  }
  const gaps: [number, number][] = [];
  if (Array.isArray(body.gaps))
    for (const g of body.gaps as unknown[]) {
      if (!Array.isArray(g)) continue;
      const a = num(g[0]);
      const b = num(g[1]);
      if (a != null && b != null && b > a) gaps.push([a * 1000, b * 1000]);
    }
  return points.length
    ? {
        points,
        intervalS: num(body.resolution_s) ?? num(body.interval_s),
        gaps,
      }
    : null;
}

/** The engine's own history for the last hour; null when the server has none. */
export async function fetchServerHistory(
  connection: Connection,
  options: {
    signal?: AbortSignal;
    now?: number;
    windowS?: number;
    binary?: boolean;
  } = {},
): Promise<ServerHistory | null> {
  const now = options.now ?? Date.now();
  try {
    const payload = await requestJson<unknown>(connection, "/yunshu/history", {
      signal: options.signal,
      timeoutMs: 8_000,
      search: { since: String(now / 1000 - (options.windowS ?? 3600)) },
    });
    return parseServerHistory(payload, options.binary);
  } catch (error) {
    if (options.signal?.aborted) throw error;
    return null;
  }
}

/**
 * The persistent history (`GET /v1/yunshu/metrics/history`): recorded from startup whether or not
 * the console was open, 1 s for the last hour, 10 s for 24 h, 1 min beyond. Null when the server
 * does not have the route (an older engine), so the caller can fall back to the in-memory ring.
 */
export async function fetchMetricsHistory(
  connection: Connection,
  options: {
    signal?: AbortSignal;
    since: number;
    until?: number;
    step?: number;
  },
): Promise<ServerHistory | null> {
  try {
    const search: Record<string, string> = {
      since: String(options.since / 1000),
    };
    if (options.until != null) search.until = String(options.until / 1000);
    if (options.step != null) search.step = String(options.step);
    const payload = await requestJson<unknown>(
      connection,
      "/yunshu/metrics/history",
      {
        signal: options.signal,
        timeoutMs: 15_000,
        search,
      },
    );
    return parseServerHistory(payload, true);
  } catch (error) {
    if (options.signal?.aborted) throw error;
    return null;
  }
}

/** What the console process says about its own view of the engine (`GET /v1/yunshu/console`). */
export interface ConsoleState {
  recording: boolean;
  up: boolean | null;
  /** Epoch ms when the engine's current state (up or down) began. */
  since: number | null;
  lastError: string | null;
}

export async function fetchConsoleState(
  connection: Connection,
  signal?: AbortSignal,
): Promise<ConsoleState | null> {
  try {
    const body = (await requestJson<Record<string, unknown>>(
      connection,
      "/yunshu/console",
      { signal, timeoutMs: 3_000 },
    )) as Record<string, unknown>;
    if (body.object !== "yunshu.console") return null;
    const since = num(body.since);
    return {
      recording: body.recording === true,
      up: typeof body.up === "boolean" ? body.up : null,
      since: since == null ? null : since * 1000,
      lastError: typeof body.last_error === "string" ? body.last_error : null,
    };
  } catch {
    // Not served by a console process (the engine directly), or unreachable: nothing to add.
    return null;
  }
}
