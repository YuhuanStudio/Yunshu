export interface TimedPoint {
  /** Epoch milliseconds. */
  t: number;
  v: number;
}

export interface SparkGeometry {
  /** Line up to the second-newest point; x is in viewBox units RELATIVE TO THE NEWEST POINT (<= 0). */
  line: string;
  /** Same line closed down to the baseline, for the area fill. */
  area: string;
  /** The newest segment (previous point to newest), drawn apart so it can fade in. */
  tail: string;
  /** viewBox units per millisecond: the time axis scale. */
  pxPerMs: number;
  /** The upper bound the values were scaled against. */
  max: number;
}

/**
 * Geometry for a time-true sparkline. The x axis is wall-clock time: a point's distance from the
 * newest point is its age difference times `pxPerMs`, so the whole line can slide left smoothly by a
 * transform while real time passes, and a new sample just extends it on the right. Points older than
 * `windowMs` before the newest one are dropped. Values are scaled to `max` (defaults to the largest
 * value, floor 1) with 0 at the bottom; y grows downwards like SVG.
 */
export function sparkGeometry(
  points: readonly TimedPoint[],
  windowMs: number,
  width: number,
  height: number,
  max?: number,
): SparkGeometry | null {
  if (points.length < 2) return null;
  const newest = points[points.length - 1].t;
  const kept = points.filter((p) => p.t >= newest - windowMs);
  if (kept.length < 2) return null;
  const top = max ?? Math.max(1, ...kept.map((p) => p.v));
  const pxPerMs = width / windowMs;
  const pad = 1;
  const y = (v: number) =>
    pad + (height - 2 * pad) * (1 - Math.min(1, Math.max(0, v / top)));
  const x = (t: number) => (t - newest) * pxPerMs;
  const fmt = (n: number) => Math.round(n * 100) / 100;
  const pts = kept.map((p) => `${fmt(x(p.t))},${fmt(y(p.v))}`);
  const full = "M" + pts.join("L");
  // the line stops at the previous sample; the newest segment is `tail`, so it can fade in
  const line = "M" + pts.slice(0, -1).join("L");
  const first = x(kept[0].t);
  const area = `${full}L0,${height}L${fmt(first)},${height}Z`;
  const prev = kept[kept.length - 2];
  const last = kept[kept.length - 1];
  const tail = `M${fmt(x(prev.t))},${fmt(y(prev.v))}L0,${fmt(y(last.v))}`;
  return { line, area, tail, pxPerMs, max: top };
}

/** How far to slide the line left, in viewBox units, `ageMs` after its newest sample arrived. */
export const slide = (ageMs: number, pxPerMs: number): number =>
  -Math.max(0, ageMs) * pxPerMs;
