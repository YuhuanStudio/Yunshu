/**
 * Rule-based reading of one finished request: where did the time go, and which requests
 * of a set are slow. Pure functions over the server's own numbers, no I/O and no wording
 * (the caller maps a `Cause` onto i18n sentences).
 *
 * Thresholds (all documented here, nothing hidden):
 * - a request under FAST_MS in total is "fast" and gets no diagnosis;
 * - a stage (queue / prefill / decode) is the cause when it is the longest of the three and
 *   takes at least DOMINANT_SHARE of the total, and lasts at least MIN_STAGE_MS;
 * - prefill is "cache miss" when at least MISS_FRESH_TOKENS prompt tokens were computed
 *   fresh; when the cache reload itself took at least half of the prefill it is a reload;
 * - decode is "slow" below SLOW_DECODE_TPS, and speculative decoding is blamed when its
 *   acceptance rate is under LOW_ACCEPT;
 * - the slow filter uses p90 of TTFT and of total once SLOW_MIN_N finished requests report
 *   a number (percentiles only from n >= 20), else the fixed SLOW_TTFT_MS / SLOW_TOTAL_MS.
 */

export const FAST_MS = 1500;
export const DOMINANT_SHARE = 0.5;
export const MIN_STAGE_MS = 1000;
export const MISS_FRESH_TOKENS = 1000;
export const SLOW_DECODE_TPS = 15;
export const LOW_ACCEPT = 0.4;
export const SLOW_MIN_N = 20;
export const SLOW_TTFT_MS = 5000;
export const SLOW_TOTAL_MS = 30000;

/** The fields of a request this module reads (a structural subset of `Row`). */
export type Timed = {
  outcome?: string;
  status_code?: number | null;
  finish_reason?: string | null;
  prompt_tokens?: number | null;
  cached_tokens?: number | null;
  completion_tokens?: number | null;
  ttft_ms?: number | null;
  decode_tps?: number | null;
  queue_wait_ms?: number | null;
  offsets_ms?: {
    admit?: number | null;
    first_token?: number | null;
    last_token?: number | null;
    done?: number | null;
  } | null;
  cache?: { tier?: string | null; reload_ms?: number | null } | null;
  speculative?: { mode?: string; acceptance_rate?: number | null } | null;
};

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

/** Server-measured pieces in ms; null where the engine did not report them. */
export function stages(r: Timed) {
  const o = r.offsets_ms ?? {};
  const admit = num(o.admit),
    first = num(o.first_token),
    last = num(o.last_token) ?? num(o.done),
    total = num(o.done);
  const queue = admit ?? num(r.queue_wait_ms);
  const ttft = num(r.ttft_ms) ?? first;
  const prefill =
    admit != null && first != null
      ? Math.max(first - admit, 0)
      : ttft != null && queue != null
        ? Math.max(ttft - queue, 0)
        : null;
  const decode =
    first != null && last != null ? Math.max(last - first, 0) : null;
  return { queue, prefill, decode, ttft, total };
}

export type Cause =
  | { kind: "unknown" }
  | { kind: "error"; code: number | null; reason: string | null }
  | { kind: "cancelled"; totalMs: number | null }
  | { kind: "fast"; totalMs: number }
  | { kind: "queue"; ms: number; share: number }
  | { kind: "prefillMiss"; ms: number; fresh: number; hitPercent: number }
  | { kind: "prefillReload"; ms: number; reloadMs: number; tier: string | null }
  | { kind: "prefill"; ms: number; fresh: number }
  | { kind: "decodeSpec"; ms: number; acceptPercent: number }
  | { kind: "decodeSlow"; ms: number; tps: number }
  | { kind: "decodeLong"; ms: number; tokens: number; tps: number | null }
  | { kind: "balanced"; totalMs: number };

/** One-sentence cause, derived only from numbers the server reported. */
export function diagnose(r: Timed): Cause {
  if (r.outcome === "error")
    return {
      kind: "error",
      code: num(r.status_code),
      reason: r.finish_reason ?? null,
    };
  const s = stages(r);
  if (r.outcome === "cancelled") return { kind: "cancelled", totalMs: s.total };
  const total = s.total ?? (s.ttft != null ? s.ttft : null);
  if (total == null) return { kind: "unknown" };
  const parts = [
    ["queue", s.queue] as const,
    ["prefill", s.prefill] as const,
    ["decode", s.decode] as const,
  ].filter(
    (p): p is readonly ["queue" | "prefill" | "decode", number] => p[1] != null,
  );
  if (total < FAST_MS) return { kind: "fast", totalMs: total };
  if (!parts.length) return { kind: "unknown" };
  const [name, ms] = parts.reduce((a, b) => (b[1] > a[1] ? b : a));
  const share = ms / total;
  if (ms < MIN_STAGE_MS || share < DOMINANT_SHARE)
    return { kind: "balanced", totalMs: total };
  const prompt = num(r.prompt_tokens) ?? 0,
    cached = Math.min(num(r.cached_tokens) ?? 0, prompt),
    fresh = prompt - cached;
  if (name === "queue") return { kind: "queue", ms, share };
  if (name === "prefill") {
    const reload = num(r.cache?.reload_ms);
    if (reload != null && reload >= ms / 2)
      return {
        kind: "prefillReload",
        ms,
        reloadMs: reload,
        tier: r.cache?.tier ?? null,
      };
    if (fresh >= MISS_FRESH_TOKENS)
      return {
        kind: "prefillMiss",
        ms,
        fresh,
        hitPercent: prompt > 0 ? (cached / prompt) * 100 : 0,
      };
    return { kind: "prefill", ms, fresh };
  }
  const accept = num(r.speculative?.acceptance_rate);
  if (accept != null && accept < LOW_ACCEPT)
    return { kind: "decodeSpec", ms, acceptPercent: accept * 100 };
  const tps = num(r.decode_tps);
  if (tps != null && tps < SLOW_DECODE_TPS)
    return { kind: "decodeSlow", ms, tps };
  return {
    kind: "decodeLong",
    ms,
    tokens: num(r.completion_tokens) ?? 0,
    tps,
  };
}

export function percentile(values: number[], p: number): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = (sorted.length - 1) * p;
  const lo = Math.floor(rank),
    hi = Math.ceil(rank);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}

export type SlowRule = {
  ttftMs: number;
  totalMs: number;
  /** "p90" when derived from the set, "fixed" below SLOW_MIN_N samples. */
  basis: "p90" | "fixed";
};

/** Threshold for "slow" over the set of finished requests being looked at. */
export function slowRule(rows: readonly Timed[]): SlowRule {
  const ttfts = rows.flatMap((r) => num(stages(r).ttft) ?? []),
    totals = rows.flatMap((r) => num(stages(r).total) ?? []);
  const t90 = ttfts.length >= SLOW_MIN_N ? percentile(ttfts, 0.9) : null,
    d90 = totals.length >= SLOW_MIN_N ? percentile(totals, 0.9) : null;
  return {
    ttftMs: t90 ?? SLOW_TTFT_MS,
    totalMs: d90 ?? SLOW_TOTAL_MS,
    basis: t90 != null || d90 != null ? "p90" : "fixed",
  };
}

/** Slow means strictly above the TTFT or the total threshold (a missing number never counts). */
export function isSlow(r: Timed, rule: SlowRule): boolean {
  const s = stages(r);
  return (
    (s.ttft != null && s.ttft > rule.ttftMs) ||
    (s.total != null && s.total > rule.totalMs)
  );
}

export type SortKey = "time" | "ttft" | "total" | "tps";
export type SortState = { key: SortKey; dir: "asc" | "desc" };

/** Sort value of a row; null sorts last in either direction. */
export function sortValue(
  r: Timed & { t?: number; t0_wall?: number | null },
  key: SortKey,
): number | null {
  if (key === "time") return num(r.t0_wall) ?? num(r.t);
  if (key === "ttft") return stages(r).ttft;
  if (key === "total") return stages(r).total;
  return num(r.decode_tps);
}

export function sortRows<
  T extends Timed & { t?: number; t0_wall?: number | null },
>(rows: readonly T[], sort: SortState): T[] {
  const sign = sort.dir === "asc" ? 1 : -1;
  return rows
    .map((row, index) => ({ row, index, v: sortValue(row, sort.key) }))
    .sort((a, b) => {
      if (a.v == null && b.v == null) return a.index - b.index;
      if (a.v == null) return 1;
      if (b.v == null) return -1;
      return a.v === b.v ? a.index - b.index : (a.v - b.v) * sign;
    })
    .map((x) => x.row);
}
