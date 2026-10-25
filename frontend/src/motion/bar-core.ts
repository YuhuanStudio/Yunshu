import { clamp01 } from "./tween-core.ts";

/**
 * A progress fraction that glides to each real sample and never past it.
 *
 * The bar is a fact (how much of the prompt the engine has processed), so it is drawn at most
 * at the latest sample: no extrapolation from a rate. Between two real samples it eases toward
 * the newer one (`settle` seconds time constant); a sample below the drawn value lowers it at
 * once, because showing more than the engine reported would be wrong.
 */
export class BarModel {
  private sample: number;
  private sampleAt: number;
  private drawn: number;
  private lastFrame: number;

  private readonly settle: number;

  /** `settle`: the time constant (s) with which the drawn value closes on the latest sample. */
  constructor(value: number, now: number, settle = 0.06) {
    this.sample = this.drawn = clamp01(value);
    this.sampleAt = this.lastFrame = now;
    this.settle = settle;
  }

  /** A new real sample at `now` (ms). */
  push(value: number, now: number): void {
    this.sample = clamp01(value);
    this.sampleAt = now;
    if (this.drawn > this.sample) this.drawn = this.sample;
  }

  /** Another request or phase: show `value` at once. */
  reset(value: number, now: number): void {
    this.sample = this.drawn = clamp01(value);
    this.sampleAt = this.lastFrame = now;
  }

  /** The fraction to draw at frame time `now`: the latest sample, eased toward, never past. */
  frame(now: number): number {
    const dt = Math.max(0, (now - this.lastFrame) / 1000);
    this.lastFrame = now;
    if (this.sample > this.drawn) {
      const k = 1 - Math.exp(-dt / this.settle);
      this.drawn = Math.min(
        this.sample,
        this.drawn + (this.sample - this.drawn) * k,
      );
    }
    return this.drawn;
  }

  get settled(): boolean {
    return Math.abs(this.drawn - this.sample) < 0.0005;
  }
}
