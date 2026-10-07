import type { EngineHistoryPoint } from "./useEngine";
import type { SeriesPoint } from "./series";
import type { EngineLastRequest, EngineStatus } from "./api";

export interface ObservedRequest extends EngineLastRequest {
  firstObservedAt: number;
}
export interface LatencyBucket {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
}
export const phaseNames: Record<string, string> = {
  queued: "排隊",
  starting: "準備中",
  prefill: "預填",
  decode: "解碼",
};
export const phaseTones = {
  queued: "warning",
  starting: "neutral",
  prefill: "info",
  decode: "success",
} as const;

/** A sampled last-request field is not a full request log. Deduplicate before any aggregation. */
export function observedRequests(
  history: readonly EngineHistoryPoint[],
): ObservedRequest[] {
  const records = new Map<string, ObservedRequest>();
  for (const sample of history) {
    const last = sample.status.last;
    if (last && !records.has(last.request_id))
      records.set(last.request_id, { ...last, firstObservedAt: sample.at });
  }
  return [...records.values()].sort(
    (a, b) => a.firstObservedAt - b.firstObservedAt,
  );
}

export function latencyDistribution(
  records: readonly ObservedRequest[],
): LatencyBucket[] {
  const limits = [0, 250, 500, 1000, 2000, 5000, Infinity];
  const labels = ["<250", "250–500", "500–1k", "1–2k", "2–5k", "≥5k"];
  const values = records.flatMap((r) =>
    typeof r.ttft_ms === "number" &&
    Number.isFinite(r.ttft_ms) &&
    r.ttft_ms >= 0
      ? [r.ttft_ms]
      : [],
  );
  if (!values.length) return [];
  return labels.map((label, i) => ({
    id: `latency-${i}`,
    label,
    value: values.filter((v) => v >= limits[i] && v < limits[i + 1]).length,
    min: limits[i],
    max: limits[i + 1],
  }));
}

/** Nearest-rank percentile of finite, nonnegative samples; unknown stays null. */
export function percentile(
  values: readonly (number | null | undefined)[],
  p: number,
): number | null {
  const sorted = values
    .filter(
      (v): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0,
    )
    .sort((a, b) => a - b);
  if (!sorted.length) return null;
  return sorted[
    Math.max(0, Math.ceil(Math.max(0, Math.min(1, p)) * sorted.length) - 1)
  ];
}

/** Percentiles from fewer samples than this are noise; show the single value instead. */
export const MIN_PERCENTILE_SAMPLES = 20;

/** `percentile`, or null while there are fewer than `MIN_PERCENTILE_SAMPLES` samples. */
export function percentileWhenEnough(
  values: readonly (number | null | undefined)[],
  p: number,
): number | null {
  const n = values.filter(
    (v) => typeof v === "number" && Number.isFinite(v) && v >= 0,
  ).length;
  return n >= MIN_PERCENTILE_SAMPLES ? percentile(values, p) : null;
}

export function phaseDistribution(status: EngineStatus | null) {
  if (!status) return [];
  const counts = new Map<string, number>();
  for (const item of status.requests.items)
    counts.set(item.phase, (counts.get(item.phase) ?? 0) + 1);
  return [...counts].map(([id, value]) => ({
    id,
    value,
    label: Object.hasOwn(phaseNames, id) ? phaseNames[id] : id,
    tone: Object.hasOwn(phaseTones, id)
      ? phaseTones[id as keyof typeof phaseTones]
      : ("neutral" as const),
  }));
}

/** Per-bucket peak of actual samples, not requests-per-period or interpolated traffic. */
export function activityHeatmap(
  history: readonly SeriesPoint[],
  start: number,
  end: number,
  columns = 12,
) {
  const count = Math.max(1, Math.floor(columns)),
    span = Math.max(1, end - start),
    step = span / count;
  const rows = ["活動請求", "排隊", "預填", "解碼"];
  const data: (number | null)[][] = rows.map(() =>
    Array.from({ length: count }, () => null),
  );
  const coverage = Array.from({ length: count }, () => 0);
  for (const point of history) {
    const at = point.at;
    if (at < start || at > end) continue;
    const index = Math.min(count - 1, Math.floor((at - start) / step));
    coverage[index]++;
    [
      point.active,
      point.queued,
      point.prefillRequests,
      point.decodeRequests,
    ].forEach((value, row) => {
      if (value != null && Number.isFinite(value) && value >= 0)
        data[row][index] =
          data[row][index] === null
            ? value
            : Math.max(data[row][index]!, value);
    });
  }
  return {
    rows,
    data,
    coverage,
    starts: Array.from({ length: count }, (_, i) => start + i * step),
    ends: Array.from({ length: count }, (_, i) => start + (i + 1) * step),
  };
}

export function observationCsv(history: readonly EngineHistoryPoint[]): string {
  const header = [
    "observed_at",
    "live_decode_tps",
    "mean_decode_tps_window",
    "mean_prefill_tps_window",
    "window_s",
    "active_requests",
    "metal_active_gb",
    "metal_cache_gb",
  ];
  const rows = history.map(({ at, status: s }) => [
    new Date(at).toISOString(),
    s.throughput.live_decode_tps,
    s.throughput.mean_decode_tps,
    s.throughput.mean_prefill_tps,
    s.throughput.window_s,
    s.requests.active,
    s.memory.active_gb,
    s.memory.cache_gb,
  ]);
  const cell = (value: unknown) => {
    const text = String(value ?? "");
    return (
      '"' +
      (/^[\s]*[=+\-@]/.test(text) ? "'" + text : text).replaceAll('"', '""') +
      '"'
    );
  };
  return (
    "\uFEFF" +
    [header, ...rows].map((row) => row.map(cell).join(",")).join("\r\n")
  );
}

export interface TrendDelta {
  /** Signed percent change of the recent half against the earlier half. */
  value: number;
  /** Which way the metric moved. Independent of whether that is good. */
  direction: "up" | "down";
  /** True when the change is an improvement (direction depends on the metric). */
  positive: boolean;
  /** Calm tone: the arrow alone carries direction, no red/green. */
  neutral: true;
  /** Display text; large changes are capped as ">100%". */
  label: string;
}

const TREND_CAP = 100;

/**
 * Honest trend: mean of the later half of the observed samples against the
 * earlier half, pct = (late - early) / early * 100. Returns null unless both
 * halves have at least `minPerSide` samples, the earlier mean is positive and
 * stable (spread within `maxSpread` of the mean) and the change is at least
 * `minPercent`; a flat, thin or bursty series shows no arrow rather than an
 * invented number. Large moves display as ">100%" instead of a raw ratio.
 */
export function trendDelta(
  values: readonly number[],
  {
    lowerIsBetter = false,
    minPerSide = 10,
    minPercent = 5,
    maxSpread = 1,
  } = {},
): TrendDelta | null {
  const clean = values.filter((v) => Number.isFinite(v));
  const half = Math.floor(clean.length / 2);
  if (half < minPerSide) return null;
  const mean = (xs: readonly number[]) =>
    xs.reduce((s, x) => s + x, 0) / xs.length;
  const earlier = clean.slice(0, half),
    later = clean.slice(clean.length - half);
  const before = mean(earlier),
    after = mean(later);
  if (!(before > 0)) return null;
  const spread =
    Math.sqrt(mean(earlier.map((x) => (x - before) ** 2))) / before;
  if (spread > maxSpread) return null;
  const value = ((after - before) / before) * 100;
  if (Math.abs(value) < minPercent) return null;
  const up = value > 0;
  return {
    value,
    direction: up ? "up" : "down",
    positive: lowerIsBetter ? !up : up,
    neutral: true,
    label:
      Math.abs(value) > TREND_CAP
        ? `>${TREND_CAP}%`
        : `${Math.abs(value).toFixed(0)}%`,
  };
}

/**
 * Rolling median over the last `k` samples (shorter at the start). A series of
 * one value per request is spiky; the median shows the level without inventing
 * points or letting one outlier dominate.
 */
export function rollingMedian(values: readonly number[], k = 5): number[] {
  return values.map((_, i) => {
    const w = values.slice(Math.max(0, i - k + 1), i + 1).sort((a, b) => a - b);
    const m = w.length >> 1;
    return w.length % 2 ? w[m] : (w[m - 1] + w[m]) / 2;
  });
}
