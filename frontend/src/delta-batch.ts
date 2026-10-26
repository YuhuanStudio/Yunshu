/**
 * Collects stream deltas and applies them once per animation frame, so a reply that arrives as
 * hundreds of chunks renders (and re-parses its Markdown) at most once per frame. `flush` applies
 * what is pending at once (end of stream, error) so nothing is left behind.
 */
export function createDeltaBatcher<D>(
  merge: (pending: D | null, next: D) => D,
  apply: (batch: D) => void,
  schedule: (run: () => void) => number = (run) => requestAnimationFrame(run),
  cancel: (handle: number) => void = (handle) => cancelAnimationFrame(handle),
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
