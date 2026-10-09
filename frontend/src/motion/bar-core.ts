import { clamp01 } from "./tween-core.ts";

/**
 * A progress fraction that moves continuously between samples.
 *
 * Real samples arrive a few times a second; drawing each one as it comes makes the bar step. The
 * model keeps the last real sample and its rate (fraction per second, smoothed over recent samples)
 * and draws `sample + rate * elapsed`, so a running prefill advances every frame. Three rules keep
 * it honest: the extrapolation is capped to `maxAhead` seconds past the last sample (a stalled
 * request does not creep on), the drawn value never exceeds 1, and it never runs backwards inside
 * one request: a sample below the drawn value waits for the extrapolation to meet it. Only `reset`
 * (a new request, a new phase) may lower the bar.
 */
export class BarModel {
  private sample: number;
  private sampleAt: number;
  private rate = 0;
  private drawn: number;
  private lastFrame: number;

  private readonly maxAhead: number;
  private readonly settle: number;

  /**
   * `maxAhead`: seconds the bar may run past the last real sample. `settle`: the time constant (s)
   * with which the drawn value closes on the model.
   */
  constructor(value: number, now: number, maxAhead = 0.45, settle = 0.06) {
    this.sample = this.drawn = clamp01(value);
    this.sampleAt = this.lastFrame = now;
    this.maxAhead = maxAhead;
    this.settle = settle;
  }

  /** A new real sample at `now` (ms). */
  push(value: number, now: number): void {
    const v = clamp01(value);
    const dt = (now - this.sampleAt) / 1000;
    if (dt > 0.02 && v >= this.sample) {
      const inst = (v - this.sample) / dt;
      this.rate = this.rate === 0 ? inst : this.rate * 0.5 + inst * 0.5;
    } else if (v < this.sample) this.rate = 0;
    this.sample = v;
    this.sampleAt = now;
  }

  /** Another request or phase: show `value` at once and forget the rate. */
  reset(value: number, now: number): void {
    this.sample = this.drawn = clamp01(value);
    this.sampleAt = this.lastFrame = now;
    this.rate = 0;
  }

  /** Where the bar should be at `now`, before smoothing. */
  modelAt(now: number): number {
    const ahead = Math.min(
      Math.max(0, (now - this.sampleAt) / 1000),
      this.maxAhead,
    );
    return clamp01(this.sample + this.rate * ahead);
  }

  /** The fraction to draw at frame time `now`: monotone, smoothed, capped. */
  frame(now: number): number {
    const dt = Math.max(0, (now - this.lastFrame) / 1000);
    this.lastFrame = now;
    const goal = this.modelAt(now);
    if (goal > this.drawn) {
      const k = 1 - Math.exp(-dt / this.settle);
      this.drawn += (goal - this.drawn) * k;
    }
    return this.drawn;
  }

  get settled(): boolean {
    return this.rate === 0 && Math.abs(this.drawn - this.sample) < 0.0005;
  }
}
