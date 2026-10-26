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
