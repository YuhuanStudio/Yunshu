/** Retry schedule while the engine is unreachable: 1 s, 2 s, 4 s, 8 s, then every ~10 s, each with ±25 % jitter. */
export const BACKOFF_BASE_MS = 1_000;
export const BACKOFF_CAP_MS = 10_000;

/**
 * Delay before the next attempt after `failures` consecutive failures (1 = the first).
 * `random` is injectable (0..1) so tests are deterministic; the jitter keeps several consoles from
 * hitting a recovering engine at the same instant.
 */
export function retryDelayMs(
  failures: number,
  random: () => number = Math.random,
): number {
  const n = Math.max(1, Math.floor(failures));
  const exp = Math.min(BACKOFF_CAP_MS, BACKOFF_BASE_MS * 2 ** (n - 1));
  const jitter = 0.75 + 0.5 * Math.min(1, Math.max(0, random()));
  return Math.round(Math.min(BACKOFF_CAP_MS * 1.25, exp * jitter));
}
