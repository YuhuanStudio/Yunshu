import type { RequestRow } from "./api.ts";

/**
 * How far one request's prefill is, with the cache hit kept apart from the work done so far. The
 * engine's `percent` is the share of the part that has to be COMPUTED, and `processed_tokens` counts
 * the cached prefix too. A bar built from `processed / prompt` therefore starts part-way along on a cache
 * hit; this keeps three separate quantities so the bar can show the cached prefix as its own segment and
 * then fill in the computed part from the end of it.
 */
export interface PrefillSplit {
  prompt: number;
  cached: number;
  /** Tokens computed so far, cache excluded. */
  done: number;
  /** Tokens that have to be computed in total (prompt minus the cache hit). */
  computed: number;
  /** Progress of the computed part, 0..100; null when the engine has not said. */
  percentOfComputed: number | null;
}

const finite = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);

export function prefillSplit(
  row: Pick<
    RequestRow,
    "prompt_tokens" | "cached_tokens" | "processed_tokens" | "percent"
  >,
): PrefillSplit | null {
  const prompt =
    finite(row.prompt_tokens) && row.prompt_tokens > 0 ? row.prompt_tokens : 0;
  const cached = Math.min(
    prompt,
    finite(row.cached_tokens) && row.cached_tokens > 0 ? row.cached_tokens : 0,
  );
  const computed = Math.max(0, prompt - cached);
  let done: number | null = null;
  if (finite(row.percent) && computed > 0)
    done = (Math.min(100, Math.max(0, row.percent)) / 100) * computed;
  else if (finite(row.processed_tokens))
    done = Math.max(0, row.processed_tokens - cached);
  if (prompt === 0) {
    // No prompt size: only the engine's own percentage is left to show.
    return finite(row.percent)
      ? {
          prompt: 0,
          cached: 0,
          done: 0,
          computed: 0,
          percentOfComputed: Math.min(100, Math.max(0, row.percent)),
        }
      : null;
  }
  if (done == null) return null;
  done = Math.min(computed, done);
  return {
    prompt,
    cached,
    done,
    computed,
    percentOfComputed: computed > 0 ? (done / computed) * 100 : 100,
  };
}
