/** Ease-out cubic: fast start, soft landing; never overshoots. */
export const easeOut = (k: number): number => 1 - Math.pow(1 - clamp01(k), 3);
export const clamp01 = (k: number): number => (k < 0 ? 0 : k > 1 ? 1 : k);

/**
 * A number that glides to each new target over `ms`. A new target starts from what is on screen
 * right now (never from the old target), so back-to-back samples blend instead of snapping.
 * `jump` is for changes that are not a continuation of the same reading (another request, another
 * phase): it sets the value at once, because gliding through the values in between would show
 * numbers that never existed.
 */
export class Tween {
  private from: number | null;
  private to: number | null;
  private t0 = 0;
  private readonly ms: number;
  constructor(value: number | null, ms = 220) {
    this.from = this.to = value;
    this.ms = ms;
  }

  get target(): number | null {
    return this.to;
  }

  valueAt(now: number): number | null {
    if (this.to == null || this.from == null) return this.to;
    if (this.from === this.to) return this.to;
    const k = (now - this.t0) / this.ms;
    return k >= 1 ? this.to : this.from + (this.to - this.from) * easeOut(k);
  }

  settled(now: number): boolean {
    return (
      this.to == null ||
      this.from == null ||
      now - this.t0 >= this.ms ||
      this.from === this.to
    );
  }

  /** Glide to `value`; null (a reading that went away) clears at once. */
  retarget(value: number | null, now: number): void {
    if (value === this.to) return;
    const shown = this.valueAt(now);
    this.from = value == null ? null : shown;
    this.to = value;
    this.t0 = now;
  }

  jump(value: number | null): void {
    this.from = this.to = value;
  }
}
