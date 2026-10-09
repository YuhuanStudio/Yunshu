import { requestJson, type Connection } from "./api.ts";
import { orUnsupported } from "./admin-models-api.ts";

const rec = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);

export type CacheTierName = "ram" | "warm" | "ssd";
export type CacheTier = {
  name: string;
  usedBytes: number | null;
  capBytes: number | null;
  entries: number | null;
  hits: number | null;
  mode: string | null;
  path: string | null;
};
export type CacheEntry = {
  key: string;
  tokens: number | null;
  bytes: number | null;
  tier: string;
  hits: number;
  lastHitAgeS: number | null;
};
export type ModelCache = {
  model: string;
  error: string | null;
  tiers: CacheTier[];
  hit: number | null;
  miss: number | null;
  byTier: Record<string, number>;
  entries: CacheEntry[];
  truncated: boolean;
};
export type CacheOverview = { caches: ModelCache[]; enabled: boolean };

export function parseCache(raw: unknown): CacheOverview {
  const r = rec(raw);
  const caches = (Array.isArray(r.caches) ? r.caches : []).map((c) => {
    const x = rec(c),
      lookups = rec(x.lookups);
    return {
      model: str(x.model) ?? "",
      error: str(x.error),
      tiers: (Array.isArray(x.tiers) ? x.tiers : []).map((t) => {
        const y = rec(t);
        return {
          name: str(y.name) ?? "",
          usedBytes: num(y.used_bytes),
          capBytes: num(y.cap_bytes),
          entries: num(y.entries),
          hits: num(y.hits),
          mode: str(y.mode),
          path: str(y.path),
        };
      }),
      hit: num(lookups.hit),
      miss: num(lookups.miss),
      byTier: Object.fromEntries(
        Object.entries(rec(lookups.by_tier)).flatMap(([k, v]) =>
          num(v) == null ? [] : [[k, v as number]],
        ),
      ),
      entries: (Array.isArray(x.entries) ? x.entries : []).map((e) => {
        const y = rec(e);
        return {
          key: str(y.key) ?? "",
          tokens: num(y.tokens),
          bytes: num(y.bytes),
          tier: str(y.tier) ?? "",
          hits: num(y.hits) ?? 0,
          lastHitAgeS: num(y.last_hit_age_s),
        };
      }),
      truncated: x.entries_truncated === true,
    };
  });
  return { caches, enabled: r.enabled === true || caches.length > 0 };
}

export const getCache = (c: Connection, signal?: AbortSignal) =>
  orUnsupported(async () =>
    parseCache(
      await requestJson<unknown>(c, "/yunshu/cache/tiers", {
        signal,
        search: { entries: "200" },
      }),
    ),
  );

export type ClearResult = { freedBytes: number | null; cleared: number };
export async function clearCache(
  c: Connection,
  body: { tier: CacheTierName | null; model: string | null },
): Promise<ClearResult> {
  const r = rec(
    await requestJson<unknown>(c, "/yunshu/cache/tiers/clear", {
      method: "POST",
      body: { tier: body.tier, model: body.model },
      timeoutMs: 60_000,
    }),
  );
  return {
    freedBytes: num(r.freed_bytes),
    cleared: Array.isArray(r.cleared) ? r.cleared.length : 0,
  };
}

/** Hit rate over a model's lookups; null until there is at least one lookup. */
export function hitRate(c: Pick<ModelCache, "hit" | "miss">): number | null {
  if (c.hit == null || c.miss == null) return null;
  const n = c.hit + c.miss;
  return n > 0 ? c.hit / n : null;
}

/** Tiers that the clear endpoint can address (ssd, ssd2... all go through "ssd"). */
export const clearableTier = (name: string): CacheTierName | null =>
  name === "ram" || name === "warm" || name === "ssd" ? name : null;
