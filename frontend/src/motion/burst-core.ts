import { reducedMotion } from "./ticker.ts";

/**
 * The rate rule for animations that data changes would start (a row arriving, a row ending, a block
 * appearing, a list making room). A monitoring screen under a burst cannot play one motion per
 * update: the eye reads it as constant twitching. So every such animation asks first, and the
 * answer is yes only while the screen is calm: at most `max` were asked for within the last
 * `windowMs`. Every ask counts, granted or not, so a storm that keeps coming stays refused until
 * it has been quiet for `windowMs`; then the next arrival animates again (a row after a quiet
 * spell, a state change on a calm screen). A refused animation is not delayed or replayed: the
 * change is simply drawn in place. Time is passed in, so tests drive it.
 */
export class BurstGovernor {
  private asks: number[] = [];
  readonly windowMs: number;
  readonly max: number;
  constructor(windowMs = 2000, max = 2) {
    this.windowMs = windowMs;
    this.max = max;
  }

  /** May an animation start now? */
  permit(now: number): boolean {
    this.asks = this.asks.filter((t) => now - t < this.windowMs);
    const calm = this.asks.length < this.max;
    this.asks.push(now);
    return calm;
  }
}

export const governor = new BurstGovernor();

/** Data-driven animations ask here: calm screen and motion allowed. */
export const mayAnimate = (): boolean =>
  !reducedMotion() && governor.permit(performance.now());
