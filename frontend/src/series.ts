import type { EngineStatus } from "./api";

/**
 * One slim chart sample. Live polls and engine-side history rows both reduce to
 * this shape, so the charts never need a full status object per point. A null
 * field means "not reported" (the engine history has no per-phase request
 * counts, for example); it is never filled in.
 */
export interface SeriesPoint {
  /** Epoch milliseconds. */
  at: number;
  /** 解碼 即時合計 (tok/s): the sum over requests that are decoding now. */
  decode: number | null;
  /** 預填 即時合計 (tok/s): the sum over requests that are prefilling now. */
  prefill: number | null;
  active: number | null;
  queued: number | null;
  prefillRequests: number | null;
  decodeRequests: number | null;
  /** Metal 記憶體 活躍 / 保留池, GB. */
  memActive: number | null;
  memCache: number | null;
  /** True for rows that came from the engine's own history, not a live poll. */
  backfilled?: boolean;
}

const finite = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

/** Sum of the per-request prefill speeds of requests that are prefilling now. */
export function livePrefillTps(status: EngineStatus): number | null {
  let sum = 0;
  let any = false;
  for (const row of status.requests.items) {
    if (row.phase !== "prefill" && row.phase !== "starting") continue;
    const tps = finite(row.tokens_per_second);
    if (tps != null) {
      sum += tps;
      any = true;
    }
  }
  return any ? sum : null;
}

export function pointFromStatus(at: number, status: EngineStatus): SeriesPoint {
  return {
    at,
    decode: finite(status.throughput.live_decode_tps),
    prefill: livePrefillTps(status),
    active: status.requests.active,
    queued: status.requests.queued,
    prefillRequests: status.requests.prefill,
    decodeRequests: status.requests.decode,
    memActive: finite(status.memory.active_gb),
    memCache: finite(status.memory.cache_gb),
  };
}

/**
 * Engine-side rows first, then live polls. Rows at or after the first live
 * sample are dropped (the live poll is the newer, richer record), and rows from
 * the future (clock skew) are ignored.
 */
export function mergeSeries(
  backfill: readonly SeriesPoint[],
  live: readonly SeriesPoint[],
  now: number,
): SeriesPoint[] {
  const firstLive = live.length ? live[0].at : now;
  const older = backfill.filter((p) => p.at < firstLive && p.at <= now);
  return older.length ? [...older, ...live] : [...live];
}

/** Index of the first point with `at >= t` (points are sorted by `at`). */
export function lowerBound(points: readonly SeriesPoint[], t: number): number {
  let lo = 0;
  let hi = points.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (points[mid].at < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Points with `start <= at <= end`, by binary search rather than a scan. */
export function windowPoints(
  points: readonly SeriesPoint[],
  start: number,
  end: number,
): SeriesPoint[] {
  const from = lowerBound(points, start);
  const to = lowerBound(points, end + 1);
  return points.slice(from, to);
}

export interface ChartRow {
  x: number;
  values: Record<string, number | null>;
}

const rowCache = new WeakMap<SeriesPoint, ChartRow>();

/** The chart row of one point, built once and reused on every later render. */
export function chartRow(point: SeriesPoint): ChartRow {
  let row = rowCache.get(point);
  if (!row) {
    row = {
      x: point.at,
      values: {
        decode: point.decode,
        prefill: point.prefill,
        active: point.memActive,
        cache: point.memCache,
        requests: point.active,
        queued: point.queued,
        prefillRequests: point.prefillRequests,
        decodeRequests: point.decodeRequests,
      },
    };
    rowCache.set(point, row);
  }
  return row;
}

/** Display rows for the charts: at most `max` of them, newest points kept exact. */
export function chartRows(
  points: readonly SeriesPoint[],
  max = 300,
): ChartRow[] {
  if (points.length <= max) return points.map(chartRow);
  const size = Math.ceil(points.length / max);
  const rows: ChartRow[] = [];
  for (let i = 0; i < points.length; i += size) {
    const group = points.slice(i, i + size);
    if (group.length === 1) {
      rows.push(chartRow(group[0]));
      continue;
    }
    const mean = (pick: (p: SeriesPoint) => number | null) => {
      let sum = 0;
      let n = 0;
      for (const p of group) {
        const v = pick(p);
        if (v != null) {
          sum += v;
          n++;
        }
      }
      return n ? sum / n : null;
    };
    const peak = (pick: (p: SeriesPoint) => number | null) => {
      let best: number | null = null;
      for (const p of group) {
        const v = pick(p);
        if (v != null && (best == null || v > best)) best = v;
      }
      return best;
    };
    rows.push({
      x: group[group.length - 1].at,
      values: {
        decode: mean((p) => p.decode),
        prefill: mean((p) => p.prefill),
        active: mean((p) => p.memActive),
        cache: mean((p) => p.memCache),
        requests: peak((p) => p.active),
        queued: peak((p) => p.queued),
        prefillRequests: peak((p) => p.prefillRequests),
        decodeRequests: peak((p) => p.decodeRequests),
      },
    });
  }
  return rows;
}
