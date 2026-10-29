/** The fastest a number meant to be READ changes on screen: about 4 Hz. Charts and bars run faster. */
export const READOUT_MS = 240;

/**
 * Pace for a text readout. A sample is drawn at once when the last draw is at least `minMs` old,
 * otherwise the newest sample waits for the end of the interval (older ones are dropped, never
 * drawn late). `now` is passed in so tests drive it.
 */
export class Readout<T> {
  private last = -Infinity;
  private pending: { value: T } | null = null;
  readonly minMs: number;
  constructor(minMs = READOUT_MS) {
    this.minMs = minMs;
  }

  /** A new sample: the value to draw now, or null when it has to wait (see `dueIn`). */
  offer(value: T, now: number): T | null {
    if (now - this.last >= this.minMs) {
      this.last = now;
      this.pending = null;
      return value;
    }
    this.pending = { value };
    return null;
  }

  /** Draw immediately and restart the interval (another request, another phase). */
  jump(value: T, now: number): T {
    this.last = now;
    this.pending = null;
    return value;
  }

  /** The waiting sample, once its interval is over. */
  flush(now: number): T | null {
    if (!this.pending || now - this.last < this.minMs) return null;
    const { value } = this.pending;
    this.pending = null;
    this.last = now;
    return value;
  }

  /** Milliseconds until the waiting sample may be drawn, or null when none is waiting. */
  dueIn(now: number): number | null {
    return this.pending ? Math.max(0, this.last + this.minMs - now) : null;
  }
}
