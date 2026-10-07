import { requestJson, type Connection } from "./api.ts";
import type { SeriesPoint } from "./series.ts";

export interface ServerHistory {
  points: SeriesPoint[];
  intervalS: number | null;
}

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

/**
 * Parse `GET /v1/yunshu/history` (columnar rows, epoch seconds). Anything that
 * is not the documented shape yields null, so an older or odd server simply
 * means "no engine-side history" and the charts start from live polls.
 */
export function parseServerHistory(payload: unknown): ServerHistory | null {
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
      decode: num(decode[i]),
      prefill: num(prefill[i]),
      active: num(active[i]),
      queued: num(queued[i]),
      prefillRequests: null,
      decodeRequests: null,
      memActive: num(memActive[i]),
      memCache: num(memCache[i]),
      backfilled: true,
    });
  }
  return points.length ? { points, intervalS: num(body.interval_s) } : null;
}

/** The engine's own history for the last hour; null when the server has none. */
export async function fetchServerHistory(
  connection: Connection,
  options: { signal?: AbortSignal; now?: number; windowS?: number } = {},
): Promise<ServerHistory | null> {
  const now = options.now ?? Date.now();
  try {
    const payload = await requestJson<unknown>(connection, "/yunshu/history", {
      signal: options.signal,
      timeoutMs: 8_000,
      search: { since: String(now / 1000 - (options.windowS ?? 3600)) },
    });
    return parseServerHistory(payload);
  } catch (error) {
    if (options.signal?.aborted) throw error;
    return null;
  }
}
