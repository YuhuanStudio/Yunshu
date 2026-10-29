/**
 * The width rule for a live reading that has to hug its content yet not follow every tick.
 *
 * The box is as wide as its widest content over the last `holdMs`: it grows the moment the content
 * needs more room (a new digit, a longer phase word) and shrinks only after the content has been
 * narrower for the whole hold, and then only down to the widest it was during that hold. A value
 * that flickers around a digit boundary (99 / 100, 9.9 / 10.1) therefore settles on the wider width
 * instead of resizing with every sample; a value that really changed for good gives its room back
 * after `holdMs`, so a settled pill carries no dead space. A shrink that is undone soon after (the
 * content needed the room again within the hold) doubles the hold, up to 8x, so a reading that keeps
 * returning to its wide form stops giving the room away; a shrink that is not undone for a long
 * while restores the base hold. Time is passed in, so tests drive it.
 */
export class WidthRatchet {
  private current: number | null = null;
  private below = false;
  private since = 0;
  private peak = 0;
  private hold: number;
  private shrunkAt = -Infinity;
  readonly holdMs: number;
  readonly tol: number;
  constructor(holdMs = 3000, tol = 0.5) {
    this.holdMs = holdMs;
    this.hold = holdMs;
    this.tol = tol;
  }

  get width(): number | null {
    return this.current;
  }

  /** Feed the content's natural width; returns the width the box should have now. */
  observe(natural: number, now: number): number {
    if (this.current == null || natural > this.current + this.tol) {
      if (this.current != null) {
        const since = now - this.shrunkAt;
        if (since < this.hold)
          this.hold = Math.min(this.hold * 2, this.holdMs * 8);
        else if (since > this.hold * 4) this.hold = this.holdMs;
      }
      this.current = natural;
      this.below = false;
      return natural;
    }
    if (natural >= this.current - this.tol) {
      this.below = false;
      return this.current;
    }
    if (!this.below) {
      this.below = true;
      this.since = now;
      this.peak = natural;
    } else this.peak = Math.max(this.peak, natural);
    return this.settle(now);
  }

  /** Apply a pending shrink whose hold has run out; returns the width to use. */
  settle(now: number): number {
    if (this.current != null && this.below && now - this.since >= this.hold) {
      this.current = this.peak;
      this.shrunkAt = now;
      this.below = false;
    }
    return this.current ?? 0;
  }

  /** Milliseconds until a pending shrink is due, or null when nothing is waiting. */
  dueIn(now: number): number | null {
    return this.below ? Math.max(0, this.since + this.hold - now) : null;
  }
}
