import { t } from "./i18n/index.ts";
import { number } from "./i18n/format.ts";

/** The slice of a finished request the prefix-reuse KPI reads (a `Row` and a recorded entry both fit). */
export interface HitSource {
  prompt_tokens?: number | null;
  cached_tokens?: number | null;
}

export interface PrefixHit {
  /** Cached tokens over prompt tokens of every request in the window, 0-100; null with no prompt tokens. */
  pct: number | null;
  /** Finished requests in the window (with a prompt). */
  count: number;
  /** The newest request's own hit rate, 0-100; null when it had no prompt. */
  latest: number | null;
}

/**
 * The one definition of "prefix hit rate" every page uses: token-weighted over the finished requests
 * the console holds (the server's recorded ring and log), newest last. Different pages, same window.
 */
export function prefixHit(rows: readonly HitSource[]): PrefixHit {
  let prompt = 0,
    cached = 0,
    count = 0,
    latest: number | null = null;
  for (const r of rows) {
    const p = r.prompt_tokens ?? 0;
    if (p <= 0) continue;
    count += 1;
    prompt += p;
    cached += r.cached_tokens ?? 0;
    latest = ((r.cached_tokens ?? 0) / p) * 100;
  }
  return { pct: prompt > 0 ? (cached / prompt) * 100 : null, count, latest };
}

/** The window line under the KPI: states what it is weighted over, identically on every page. */
export function prefixHitWindow(hit: PrefixHit, recorded: boolean): string {
  if (hit.pct == null) return t("requests.tile.noFinished");
  return (
    t(recorded ? "requests.tile.hitServer" : "requests.tile.hitPage", {
      count: number(hit.count, 0),
    }) +
    (hit.latest != null
      ? t("requests.tile.hitLatest", { value: number(hit.latest, 0) })
      : "")
  );
}
