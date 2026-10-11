/**
 * One requestAnimationFrame loop for every live number, bar and chart. Subscribers run once per
 * frame with the frame time; the loop sleeps when nobody is subscribed. Nothing here touches React:
 * subscribers write to the DOM (text, transform, CSS variables) themselves, so a frame costs no
 * re-render. The clock is injectable, so tests drive time by hand.
 */
export interface Clock {
  now(): number;
  request(cb: (t: number) => void): number;
  cancel(id: number): void;
}

export const browserClock: Clock = {
  now: () => performance.now(),
  request: (cb) => requestAnimationFrame(cb),
  cancel: (id) => cancelAnimationFrame(id),
};

export type Frame = (now: number) => void;

export class Ticker {
  private subs = new Set<Frame>();
  private id: number | null = null;
  readonly clock: Clock;
  constructor(clock: Clock = browserClock) {
    this.clock = clock;
  }

  get size(): number {
    return this.subs.size;
  }
  get running(): boolean {
    return this.id !== null;
  }

  subscribe(fn: Frame): () => void {
    this.subs.add(fn);
    if (this.id === null) this.id = this.clock.request(this.frame);
    return () => {
      this.subs.delete(fn);
    };
  }

  private frame = (t: number): void => {
    this.id = null;
    if (!this.subs.size) return;
    for (const fn of [...this.subs]) fn(t);
    if (this.subs.size) this.id = this.clock.request(this.frame);
  };
}

export const ticker = new Ticker();

/** prefers-reduced-motion: reduce. Read on every use, so a change in system settings applies live. */
export const reducedMotion = (): boolean =>
  typeof window !== "undefined" &&
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;
