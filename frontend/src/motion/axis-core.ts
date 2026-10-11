/**
 * The value axis of a live chart must not rescale on every sample: it holds a "nice" maximum that
 * grows at once when a value exceeds it and shrinks only after the data has stayed below SHRINK_BELOW
 * of it for SHRINK_AFTER_MS. Pure, so tests drive it with a fake clock.
 */
export const SHRINK_BELOW = 0.6;
export const SHRINK_AFTER_MS = 8_000;

/** The smallest of 1, 2, 2.5, 5, 10 x 10^k that is at least `value` (a minimum of 1). */
export function niceCeil(value: number): number {
  if (!Number.isFinite(value) || value <= 1) return 1;
  const exp = Math.floor(Math.log10(value));
  const base = 10 ** exp;
  for (const m of [1, 2, 2.5, 5, 10])
    if (value <= m * base + 1e-9) return m * base;
  return 10 * base;
}

export interface AxisState {
  max: number;
  /** When the data first stayed below the shrink threshold; null while it has not. */
  lowSince: number | null;
}

export function initialAxis(dataMax: number): AxisState {
  return { max: niceCeil(dataMax), lowSince: null };
}

/** The axis after seeing `dataMax` at time `now` (ms). */
export function nextAxis(
  state: AxisState,
  dataMax: number,
  now: number,
): AxisState {
  if (dataMax > state.max)
    return { max: niceCeil(dataMax), lowSince: null };
  const target = niceCeil(dataMax);
  if (target < state.max && dataMax < state.max * SHRINK_BELOW) {
    const since = state.lowSince ?? now;
    if (now - since >= SHRINK_AFTER_MS) return { max: target, lowSince: null };
    return { max: state.max, lowSince: since };
  }
  return state.lowSince == null ? state : { max: state.max, lowSince: null };
}

/**
 * Absolute tick times (epoch ms) inside `[from, to]`, on multiples of a round step, about `count` of
 * them. Absolute, so a tick is the same tick while the window slides and only labels enter and leave.
 */
export function timeTicks(from: number, to: number, count: number): number[] {
  const span = Math.max(1, to - from);
  const want = span / Math.max(1, count);
  const steps = [
    1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 21600, 43200,
    86400,
  ].map((s) => s * 1000);
  const step = steps.find((s) => s >= want) ?? steps[steps.length - 1];
  const out: number[] = [];
  for (let t = Math.ceil(from / step) * step; t <= to; t += step) out.push(t);
  return out;
}
