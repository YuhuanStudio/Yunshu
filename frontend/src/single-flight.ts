/**
 * Runs `run` one at a time. A trigger that arrives while it is running is not dropped: it queues one
 * more run after the current one (however many arrive meanwhile), so a request made after a fetch
 * started is never answered by that earlier fetch alone. The returned promise settles when the run
 * that covers this trigger has finished.
 */
export function createSingleFlight(run: () => Promise<void>) {
  let current: Promise<void> | null = null;
  let queued = false;
  return function trigger(): Promise<void> {
    if (current) {
      queued = true;
      return current;
    }
    current = (async () => {
      try {
        do {
          queued = false;
          await run();
        } while (queued);
      } finally {
        current = null;
      }
    })();
    return current;
  };
}

/** A signal that aborts with the caller's `signal` or after `ms`, so one hung response cannot stall a poller. */
export function withTimeout(signal: AbortSignal, ms: number): AbortSignal {
  // AbortSignal.any / .timeout need Safari 17.4 / Chrome 116; older browsers get the same thing by hand.
  if (
    typeof AbortSignal.any === "function" &&
    typeof AbortSignal.timeout === "function"
  ) {
    return AbortSignal.any([signal, AbortSignal.timeout(ms)]);
  }
  const controller = new AbortController();
  if (signal.aborted) {
    controller.abort(signal.reason);
    return controller.signal;
  }
  const timer = setTimeout(
    () => controller.abort(new DOMException("The operation timed out.", "TimeoutError")),
    ms,
  );
  signal.addEventListener(
    "abort",
    () => {
      clearTimeout(timer);
      controller.abort(signal.reason);
    },
    { once: true },
  );
  controller.signal.addEventListener("abort", () => clearTimeout(timer), {
    once: true,
  });
  return controller.signal;
}

/**
 * A poll is already in flight: a background request shares its answer, but a forced one (the user's
 * refresh, or a read after a change) needs a fetch that starts after the change, so it runs when the
 * in-flight one is done.
 */
export function joinOrQueue(
  inFlight: Promise<void> | null,
  force: boolean,
  start: () => Promise<void>,
): Promise<void> {
  if (!inFlight) return start();
  return force ? inFlight.then(start, start) : inFlight;
}
