import assert from "node:assert/strict";
import test from "node:test";
import {
  createSingleFlight,
  joinOrQueue,
  withTimeout,
} from "../src/single-flight.ts";

test("a trigger during a run queues exactly one follow-up run", async () => {
  let runs = 0;
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  const trigger = createSingleFlight(async () => {
    runs++;
    if (runs === 1) await gate;
  });
  const first = trigger();
  trigger();
  const last = trigger();
  assert.equal(runs, 1);
  release();
  await Promise.all([first, last]);
  assert.equal(runs, 2);
  await trigger();
  assert.equal(runs, 3);
});

test("withTimeout aborts a hung request", async () => {
  const keepAlive = setTimeout(() => {}, 1000);
  const s = withTimeout(new AbortController().signal, 20);
  await new Promise((r) => s.addEventListener("abort", r));
  assert.equal(s.aborted, true);
  clearTimeout(keepAlive);
});

test("a forced refresh waits for the poll in flight and then fetches again; a background one shares it", async () => {
  let starts = 0;
  const start = async () => void starts++;
  let done!: () => void;
  const inFlight = new Promise<void>((r) => (done = r));
  const shared = joinOrQueue(inFlight, false, start);
  const forced = joinOrQueue(inFlight, true, start);
  assert.equal(starts, 0);
  done();
  await Promise.all([shared, forced]);
  assert.equal(starts, 1);
  await joinOrQueue(null, false, start);
  assert.equal(starts, 2);
});

test("withTimeout works without AbortSignal.any / AbortSignal.timeout (older Safari and Chrome)", async () => {
  const any = AbortSignal.any;
  const timeout = AbortSignal.timeout;
  // @ts-expect-error simulate a browser that predates both
  AbortSignal.any = undefined;
  // @ts-expect-error simulate a browser that predates both
  AbortSignal.timeout = undefined;
  try {
    const caller = new AbortController();
    const s = withTimeout(caller.signal, 20);
    assert.equal(s.aborted, false);
    await new Promise((r) => setTimeout(r, 60));
    assert.equal(s.aborted, true);
    assert.equal((s.reason as DOMException).name, "TimeoutError");
    const c2 = new AbortController();
    const s2 = withTimeout(c2.signal, 10_000);
    c2.abort(new Error("stop"));
    assert.equal(s2.aborted, true);
    const done = new AbortController();
    done.abort();
    assert.equal(withTimeout(done.signal, 10_000).aborted, true);
  } finally {
    AbortSignal.any = any;
    AbortSignal.timeout = timeout;
  }
});
