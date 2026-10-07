import type { EngineHistoryPoint } from "./useEngine";
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
  prefill: "Prefill",
  decode: "Decode",
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

/** Compare means only with means. Null samples and collection gaps remain explicit. */
export function timeSeries(history: readonly EngineHistoryPoint[]) {
  return history.map(({ at, status }) => ({
    x: at,
    values: {
      decode: status.throughput.mean_decode_tps,
      prefill: status.throughput.mean_prefill_tps,
      active: status.memory.active_gb ?? null,
      cache: status.memory.cache_gb ?? null,
      requests: status.requests.active,
      queued: status.requests.queued,
      prefillRequests: status.requests.prefill,
      decodeRequests: status.requests.decode,
    },
  }));
}

/** Per-bucket peak of actual samples, not requests-per-period or interpolated traffic. */
export function activityHeatmap(
  history: readonly EngineHistoryPoint[],
  start: number,
  end: number,
  columns = 12,
) {
  const count = Math.max(1, Math.floor(columns)),
    span = Math.max(1, end - start),
    step = span / count;
  const rows = ["活動請求", "排隊", "Prefill", "Decode"];
  const data: (number | null)[][] = rows.map(() =>
    Array.from({ length: count }, () => null),
  );
  const coverage = Array.from({ length: count }, () => 0);
  for (const { at, status } of history) {
    if (at < start || at > end) continue;
    const index = Math.min(count - 1, Math.floor((at - start) / step));
    coverage[index]++;
    [
      status.requests.active,
      status.requests.queued,
      status.requests.prefill,
      status.requests.decode,
    ].forEach((value, row) => {
      if (Number.isFinite(value) && value >= 0)
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
    "mean_decode_tps_300s",
    "mean_prefill_tps_300s",
    "active_requests",
    "metal_active_gb",
    "metal_cache_gb",
  ];
  const rows = history.map(({ at, status: s }) => [
    new Date(at).toISOString(),
    s.throughput.mean_decode_tps,
    s.throughput.mean_prefill_tps,
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
  /** True when the change is an improvement (direction depends on the metric). */
  positive: boolean;
}

/**
 * Honest trend: mean of the later half of the observed samples against the
 * earlier half. Returns null unless both halves have enough samples, the
 * earlier mean is positive, and the change is at least `minPercent`; a flat or
 * unobserved series shows no arrow rather than an invented 0%.
 */
export function trendDelta(
  values: readonly number[],
  { lowerIsBetter = false, minPerSide = 3, minPercent = 5 } = {},
): TrendDelta | null {
  const clean = values.filter((v) => Number.isFinite(v));
  const half = Math.floor(clean.length / 2);
  if (half < minPerSide) return null;
  const mean = (xs: readonly number[]) =>
    xs.reduce((s, x) => s + x, 0) / xs.length;
  const before = mean(clean.slice(0, half)),
    after = mean(clean.slice(clean.length - half));
  if (!(before > 0)) return null;
  const value = ((after - before) / before) * 100;
  if (Math.abs(value) < minPercent) return null;
  return { value, positive: lowerIsBetter ? value < 0 : value > 0 };
}
