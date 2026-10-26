import assert from "node:assert/strict";
import test from "node:test";
import { createDeltaBatcher, createFrameScheduler } from "../src/delta-batch.ts";

test("many deltas within a frame are applied once, merged in order", () => {
  const frames: (() => void)[] = [];
  const applied: string[] = [];
  const b = createDeltaBatcher<string>(
    (p, n) => (p ?? "") + n,
    (s) => applied.push(s),
    (run) => frames.push(run),
    () => undefined,
  );
  b.push("a");
  b.push("b");
  b.push("c");
  assert.deepEqual(applied, []);
  assert.equal(frames.length, 1);
  frames[0]();
  assert.deepEqual(applied, ["abc"]);
  b.push("d");
  assert.equal(frames.length, 2);
});

test("flush applies the pending deltas immediately and a later frame has nothing left", () => {
  const frames: (() => void)[] = [];
  const applied: string[] = [];
  const b = createDeltaBatcher<string>(
    (p, n) => (p ?? "") + n,
    (s) => applied.push(s),
    (run) => frames.push(run),
    () => undefined,
  );
  b.push("x");
  b.flush();
  assert.deepEqual(applied, ["x"]);
  frames[0]?.();
  assert.deepEqual(applied, ["x"]);
});

test("the default scheduler falls back to a timer in a hidden tab", () => {
  let hidden = true;
  const timers = new Map<number, { run: () => void; ms: number }>();
  const frames = new Map<number, () => void>();
  let n = 0;
  const sched = createFrameScheduler({
    raf: (run) => (frames.set(++n, run), n),
    caf: (h) => void frames.delete(h),
    setTimer: (run, ms) => (timers.set(++n, { run, ms }), n),
    clearTimer: (h) => void timers.delete(h),
    hidden: () => hidden,
  });
  const ran: string[] = [];
  const h = sched.schedule(() => ran.push("a"));
  assert.equal(frames.size, 0, "no frame is requested in a hidden tab");
  assert.equal([...timers.values()][0].ms, 100);
  [...timers.values()][0].run();
  assert.deepEqual(ran, ["a"]);
  assert.equal(timers.size, 0);
  // visible: the frame wins and the safety timer is cleared; cancel drops both
  hidden = false;
  sched.schedule(() => ran.push("b"));
  assert.equal(frames.size, 1);
  assert.equal([...timers.values()][0].ms, 500);
  [...frames.values()][0]();
  assert.deepEqual(ran, ["a", "b"]);
  assert.equal(timers.size, 0);
  const h2 = sched.schedule(() => ran.push("c"));
  sched.cancel(h2);
  assert.equal(frames.size + timers.size, 0);
  sched.cancel(h); // an already finished handle is harmless
  // a tab that hides after the frame was requested still advances through the safety timer
  sched.schedule(() => ran.push("d"));
  hidden = true;
  [...timers.values()][0].run();
  assert.equal(ran.at(-1), "d");
  assert.equal(frames.size, 0);
});
