/**
 * Prefix-cache lifecycle from `GET /debug/kv-cache` (`caches[].apc`, the engine's APC snapshot):
 * what was admitted, moved between tiers, evicted or dropped, and what hit. The engine reports
 * counters per kind, not a reason per event, so the "reason" is the counter that moved.
 * A counter the engine did not send is absent here; a reported zero stays zero.
 */
export type Stage = "admission" | "movement" | "removal" | "hits";

export interface LifecycleCounter {
  /** The engine's counter name, also the i18n suffix. */
  id: string;
  stage: Stage;
  value: number;
}

export interface LifecycleModel {
  model: string;
  counters: LifecycleCounter[];
  /** Occupancy figures the engine reported, in bytes. */
  bytes: {
    ram: number | null;
    warm: number | null;
    warmRatio: number | null;
    ssd: number | null;
  };
  entries: number | null;
}

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;
const rec = (v: unknown): Record<string, unknown> | null =>
  v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : null;

/** Counter id -> stage, in display order. */
export const COUNTERS: ReadonlyArray<readonly [string, Stage]> = [
  ["exact_stores", "admission"],
  ["memory_skips", "admission"],
  ["warm_rejected", "admission"],
  ["cost_rejected", "admission"],
  ["warm_demotions", "movement"],
  ["warm_evicted_to_ssd", "movement"],
  ["disk_writes", "movement"],
  ["memory_evictions", "removal"],
  ["warm_dropped", "removal"],
  ["warm_corrupt", "removal"],
  ["invalidated", "removal"],
  ["lookups_hit", "hits"],
  ["lookups_miss", "hits"],
  ["exact_hits", "hits"],
  ["warm_hits", "hits"],
  ["disk_hits", "hits"],
  ["matched_tokens", "hits"],
];

/** Sum of a per-tier counter across `storage_tiers`; null when no tier reports it. */
function tierSum(tiers: unknown, key: string): number | null {
  if (!Array.isArray(tiers)) return null;
  let sum: number | null = null;
  for (const t of tiers) {
    const v = num(rec(t)?.[key]);
    if (v != null) sum = (sum ?? 0) + v;
  }
  return sum;
}

export function parseLifecycle(raw: unknown): LifecycleModel[] {
  const caches = rec(raw)?.caches;
  if (!Array.isArray(caches)) return [];
  const out: LifecycleModel[] = [];
  for (const c of caches) {
    const x = rec(c);
    const apc = rec(x?.apc);
    if (!x || !apc) continue;
    const counters: LifecycleCounter[] = [];
    for (const [id, stage] of COUNTERS) {
      const v =
        id === "cost_rejected" || id === "invalidated"
          ? tierSum(apc.storage_tiers, id)
          : num(apc[id]);
      if (v != null) counters.push({ id, stage, value: v });
    }
    out.push({
      model: typeof x.model_id === "string" ? x.model_id : "",
      counters,
      bytes: {
        ram: num(apc.resident_bytes),
        warm: num(apc.warm_bytes),
        warmRatio: num(apc.warm_ratio),
        ssd: num(apc.disk_bytes),
      },
      entries: num(apc.entries),
    });
  }
  return out;
}

/** Hit requests and matched tokens are different quantities; this is the request hit rate. */
export function requestHitRate(m: LifecycleModel): number | null {
  const hit = m.counters.find((c) => c.id === "lookups_hit")?.value;
  const miss = m.counters.find((c) => c.id === "lookups_miss")?.value;
  if (hit == null || miss == null || hit + miss === 0) return null;
  return hit / (hit + miss);
}
