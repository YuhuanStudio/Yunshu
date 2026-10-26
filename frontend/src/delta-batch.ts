export interface FrameEnv {
  raf: (run: () => void) => number;
  caf: (handle: number) => void;
  setTimer: (run: () => void, ms: number) => number;
  clearTimer: (handle: number) => void;
  hidden: () => boolean;
}

const browserEnv = (): FrameEnv => ({
  raf: (run) => requestAnimationFrame(run),
  caf: (handle) => cancelAnimationFrame(handle),
  setTimer: (run, ms) => setTimeout(run, ms) as unknown as number,
  clearTimer: (handle) => clearTimeout(handle),
  hidden: () => typeof document !== "undefined" && document.hidden,
});

/**
 * The default scheduler: the next animation frame, but a hidden tab never runs frames, so a hidden tab
 * (or one that hides while a frame is pending) falls back to a timer and the reply keeps advancing.
 */
export function createFrameScheduler(env: FrameEnv = browserEnv()) {
  const pending = new Map<number, () => void>();
  let seq = 0;
  return {
    schedule(run: () => void): number {
      const id = ++seq;
      let frame: number | null = null;
      let timer: number | null = null;
      const cancel = () => {
        if (frame !== null) env.caf(frame);
        if (timer !== null) env.clearTimer(timer);
        pending.delete(id);
      };
      const go = () => {
        cancel();
        run();
      };
      // the timer is the fallback: short when already hidden, a safety net when a frame is pending
      if (!env.hidden()) frame = env.raf(go);
      timer = env.setTimer(go, env.hidden() ? 100 : 500);
      pending.set(id, cancel);
      return id;
    },
    cancel(handle: number) {
      pending.get(handle)?.();
    },
  };
}

const defaultScheduler = createFrameScheduler();

/**
 * Collects stream deltas and applies them once per animation frame, so a reply that arrives as
 * hundreds of chunks renders (and re-parses its Markdown) at most once per frame. `flush` applies
 * what is pending at once (end of stream, error) so nothing is left behind.
 */
export function createDeltaBatcher<D>(
  merge: (pending: D | null, next: D) => D,
  apply: (batch: D) => void,
  schedule: (run: () => void) => number = defaultScheduler.schedule,
  cancel: (handle: number) => void = defaultScheduler.cancel,
) {
  let pending: D | null = null;
  let handle: number | null = null;
  const flush = () => {
    if (handle !== null) cancel(handle);
    handle = null;
    if (pending === null) return;
    const batch = pending;
    pending = null;
    apply(batch);
  };
  return {
    push(next: D) {
      pending = merge(pending, next);
      if (handle === null) handle = schedule(flush);
    },
    flush,
    /** Drops what is pending (the run was abandoned). */
    discard() {
      if (handle !== null) cancel(handle);
      handle = null;
      pending = null;
    },
  };
}
