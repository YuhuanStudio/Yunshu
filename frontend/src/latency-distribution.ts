import { MIN_PERCENTILE_SAMPLES, percentile } from "./analytics.ts";
import type { Row } from "./RequestTrace";

/** What a distribution is split by. "all" is one group. */
export type Dimension = "all" | "warmth" | "context" | "model";
export type Metric = "ttft" | "queue";

/** A request is warm when at least half of its prompt came from the prefix cache. */
export const WARM_SHARE = 0.5;
/** Context bands, in prompt tokens: the upper edge is exclusive. */
export const CONTEXT_EDGES = [1_000, 8_000, 32_000] as const;

type Fields = Pick<
  Row,
  | "ttft_ms"
  | "queue_wait_ms"
  | "prompt_tokens"
  | "cached_tokens"
  | "model"
  | "outcome"
>;

export const metricValue = (r: Fields, metric: Metric): number | null => {
  const v = metric === "ttft" ? r.ttft_ms : r.queue_wait_ms;
  return typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;
};

/** Group id for a row, or null when the row cannot be placed (unknown prompt size, no model). */
export function groupOf(r: Fields, dim: Dimension): string | null {
  switch (dim) {
    case "all":
      return "all";
    case "warmth": {
      const p = r.prompt_tokens;
      if (typeof p !== "number" || p <= 0) return null;
      const c = r.cached_tokens;
      if (typeof c !== "number") return null;
      return c / p >= WARM_SHARE ? "warm" : "cold";
    }
    case "context": {
      const p = r.prompt_tokens;
      if (typeof p !== "number" || p < 0) return null;
      const i = CONTEXT_EDGES.findIndex((e) => p < e);
      return `ctx${i === -1 ? CONTEXT_EDGES.length : i}`;
    }
    case "model":
      return r.model ? `model:${r.model}` : null;
  }
}

export interface GroupStats {
  id: string;
  /** Samples of the chosen metric in this group. */
  n: number;
  /** Null below MIN_PERCENTILE_SAMPLES: a percentile from a handful of requests is noise. */
  p50: number | null;
  p90: number | null;
}

/**
 * Per-group sample counts and percentiles of one metric. Failed requests are left out of TTFT
 * (they never produced a first token); the rest of the rows count wherever the metric is reported.
 */
export function distribution(
  rows: readonly Fields[],
  dim: Dimension,
  metric: Metric,
): { groups: GroupStats[]; unplaced: number } {
  const byGroup = new Map<string, number[]>();
  let unplaced = 0;
  for (const r of rows) {
    const v = metricValue(r, metric);
    if (v == null) continue;
    if (metric === "ttft" && r.outcome === "error") continue;
    const g = groupOf(r, dim);
    if (g == null) {
      unplaced += 1;
      continue;
    }
    const list = byGroup.get(g) ?? [];
    list.push(v);
    byGroup.set(g, list);
  }
  const groups = [...byGroup].map(([id, values]) => ({
    id,
    n: values.length,
    p50:
      values.length >= MIN_PERCENTILE_SAMPLES ? percentile(values, 0.5) : null,
    p90:
      values.length >= MIN_PERCENTILE_SAMPLES ? percentile(values, 0.9) : null,
  }));
  const order = (id: string) =>
    id === "cold"
      ? 0
      : id === "warm"
        ? 1
        : id.startsWith("ctx")
          ? Number(id.slice(3))
          : 0;
  groups.sort((a, b) =>
    dim === "model"
      ? b.n - a.n || a.id.localeCompare(b.id)
      : order(a.id) - order(b.id),
  );
  return { groups, unplaced };
}

export const BIN_EDGES = [0, 100, 250, 500, 1000, 2000, 5000, 10000, Infinity];

/** Counts per latency bucket for the whole selection; shows the shape that percentiles hide. */
export function histogram(
  rows: readonly Fields[],
  metric: Metric,
): { min: number; max: number; count: number }[] {
  const values = rows.flatMap((r) => {
    const v = metricValue(r, metric);
    return v == null || (metric === "ttft" && r.outcome === "error") ? [] : [v];
  });
  if (!values.length) return [];
  return BIN_EDGES.slice(0, -1).map((min, i) => ({
    min,
    max: BIN_EDGES[i + 1],
    count: values.filter((v) => v >= min && v < BIN_EDGES[i + 1]).length,
  }));
}
