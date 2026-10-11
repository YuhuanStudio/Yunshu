import type { CompletionUsage } from "./stream";

/** Client-side timing of one streamed reply (all times are performance.now() ms). */
export interface RunTiming {
  start: number;
  firstAt?: number;
  endAt?: number;
  /** Number of content/reasoning deltas received (a token estimate). */
  chunks: number;
  usage?: CompletionUsage;
}

export interface RunStats {
  tokens: number;
  /** True when tokens come from delta count, not server usage. */
  estimated: boolean;
  latencyMs: number;
  ttftMs?: number;
  tokensPerSecond?: number;
  cachedTokens?: number;
  promptTokens?: number;
}

/** Compute stats at `now`; decode rate excludes the wait for the first token. */
export function runStats(t: RunTiming, now: number): RunStats {
  const end = t.endAt ?? now;
  const latencyMs = Math.max(0, end - t.start);
  const tokens = t.usage?.completionTokens ?? t.chunks;
  const ttftMs =
    t.usage?.ttftMs ??
    (t.firstAt === undefined ? undefined : Math.max(0, t.firstAt - t.start));
  let tokensPerSecond: number | undefined;
  if (t.firstAt !== undefined && tokens > 1 && end > t.firstAt)
    tokensPerSecond = ((tokens - 1) * 1000) / (end - t.firstAt);
  else if (tokens > 0 && latencyMs > 0)
    tokensPerSecond = (tokens * 1000) / latencyMs;
  return {
    tokens,
    estimated: t.usage?.completionTokens === undefined,
    latencyMs,
    ttftMs,
    tokensPerSecond,
    cachedTokens: t.usage?.cachedTokens,
    promptTokens: t.usage?.promptTokens,
  };
}

/** Index of the first differing UTF-16 offset, or null when strings are equal. */
export function firstDivergence(a: string, b: string): number | null {
  if (a === b) return null;
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

export type Comparison =
  { kind: "identical"; greedy: boolean } | { kind: "diverged"; offset: number };

/** Only called with finished, non-empty replies; equality is exact string equality. */
export function compareOutputs(
  a: { text: string; temperature: number },
  b: { text: string; temperature: number },
): Comparison {
  const offset = firstDivergence(a.text, b.text);
  if (offset === null)
    return {
      kind: "identical",
      greedy: a.temperature === 0 && b.temperature === 0,
    };
  return { kind: "diverged", offset };
}
