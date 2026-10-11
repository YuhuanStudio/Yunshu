import assert from "node:assert/strict";
import test from "node:test";
import { Ticker, type Clock } from "../src/motion/ticker.ts";
import { Tween, easeOut } from "../src/motion/tween-core.ts";
import { BarModel } from "../src/motion/bar-core.ts";

/** A clock the test advances by hand: frames run only when `advance` says so. */
class FakeClock implements Clock {
  t = 0;
  private next = 1;
  private pending = new Map<number, (t: number) => void>();
  now = () => this.t;
  request = (cb: (t: number) => void) => {
    const id = this.next++;
    this.pending.set(id, cb);
    return id;
  };
  cancel = (id: number) => void this.pending.delete(id);
  get queued() {
    return this.pending.size;
  }
  advance(ms: number) {
    this.t += ms;
    const run = [...this.pending.values()];
    this.pending.clear();
    for (const cb of run) cb(this.t);
  }
}

test("ticker: one loop for all subscribers, asleep when none", () => {
  const clock = new FakeClock();
  const tk = new Ticker(clock);
  const a: number[] = [];
  const b: number[] = [];
  const offA = tk.subscribe((t) => a.push(t));
  tk.subscribe((t) => b.push(t));
  assert.equal(clock.queued, 1);
  clock.advance(16);
  clock.advance(16);
  assert.deepEqual(a, [16, 32]);
  assert.deepEqual(b, [16, 32]);
  offA();
  clock.advance(16);
  assert.deepEqual(a, [16, 32]);
  assert.equal(b.length, 3);
});

test("ticker: the loop stops when the last subscriber leaves and restarts on the next", () => {
  const clock = new FakeClock();
  const tk = new Ticker(clock);
  const off = tk.subscribe(() => {});
  off();
  clock.advance(16);
  assert.equal(tk.running, false);
  assert.equal(clock.queued, 0);
  tk.subscribe(() => {});
  assert.equal(clock.queued, 1);
});

test("tween: ease-out to the target over the duration, no overshoot", () => {
  const tw = new Tween(0, 200);
  tw.retarget(100, 1000);
  assert.equal(tw.valueAt(1000), 0);
  const mid = tw.valueAt(1100)!;
  assert.ok(mid > 50 && mid < 100, "ease-out is past the linear midpoint");
  assert.equal(tw.valueAt(1200), 100);
  assert.equal(tw.valueAt(5000), 100);
  for (let ms = 0; ms <= 300; ms += 10)
    assert.ok(tw.valueAt(1000 + ms)! <= 100);
  assert.ok(easeOut(0.5) > 0.5);
});

test("tween: a new target starts from what is on screen, not from the old target", () => {
  const tw = new Tween(0, 200);
  tw.retarget(100, 0);
  const shown = tw.valueAt(100)!;
  tw.retarget(0, 100);
  assert.equal(tw.valueAt(100), shown);
  assert.ok(tw.valueAt(150)! < shown);
});

test("tween: null clears at once; jump never shows in-between values", () => {
  const tw = new Tween(40, 200);
  tw.retarget(null, 0);
  assert.equal(tw.valueAt(10), null);
  tw.retarget(80, 20);
  assert.equal(
    tw.valueAt(20),
    80,
    "from nothing there is nothing to glide from",
  );
  tw.jump(5);
  assert.equal(tw.valueAt(21), 5);
  assert.equal(tw.settled(21), true);
});

test("bar: advances every frame between samples at the observed rate", () => {
  const bar = new BarModel(0, 0);
  bar.push(0.1, 250); // 0.4/s
  const frames: number[] = [];
  for (let t = 250; t <= 500; t += 16) frames.push(bar.frame(t));
  assert.ok(
    frames.every((v, i) => i === 0 || v >= frames[i - 1]),
    "monotone",
  );
  assert.ok(
    new Set(frames.map((v) => v.toFixed(4))).size > 8,
    "moves on (almost) every frame, not in steps",
  );
  assert.ok(frames.at(-1)! > 0.1 && frames.at(-1)! < 0.35);
});

test("bar: extrapolation is capped, so a stalled request does not creep on", () => {
  const bar = new BarModel(0, 0);
  bar.push(0.2, 250);
  let v = 0;
  for (let t = 250; t <= 5000; t += 16) v = bar.frame(t);
  assert.ok(v <= 0.2 + 0.8 * 0.45 + 1e-6);
  const later = bar.frame(10_000);
  assert.ok(Math.abs(later - v) < 1e-6);
});

test("bar: never runs backwards inside a request, never past 1", () => {
  const bar = new BarModel(0.5, 0);
  bar.push(0.7, 250);
  let last = 0;
  for (let t = 250; t <= 500; t += 16) last = bar.frame(t);
  bar.push(0.65, 500); // a late, lower sample
  assert.ok(bar.frame(516) >= last);
  const full = new BarModel(0.9, 0);
  full.push(0.99, 250);
  for (let t = 250; t <= 2000; t += 16) assert.ok(full.frame(t) <= 1);
});

test("bar: reset (a new request) lowers it at once and forgets the rate", () => {
  const bar = new BarModel(0.9, 0);
  bar.push(1, 250);
  bar.frame(300);
  bar.reset(0, 400);
  assert.equal(bar.frame(401), 0);
  assert.equal(bar.frame(900), 0, "no leftover rate");
});

import { sparkGeometry, slide } from "../src/motion/spark-core.ts";

test("spark: x is wall-clock distance from the newest point; the newest segment is apart", () => {
  const pts = [0, 1000, 2000, 3000].map((t, i) => ({
    t: 10_000 + t,
    v: [1, 3, 2, 4][i],
  }));
  const g = sparkGeometry(pts, 60_000, 600, 40)!;
  assert.equal(g.pxPerMs, 0.01);
  assert.ok(
    g.line.startsWith("M-30,"),
    "oldest point 3 s before the newest is 30 units left",
  );
  assert.ok(
    g.tail.endsWith("L0," + g.tail.split("L0,")[1]),
    "tail ends at x = 0 (the newest point)",
  );
  assert.ok(!g.line.includes("L0,"), "the line stops one sample short");
  assert.equal(g.max, 4);
});

test("spark: points older than the window drop, fewer than two draw nothing, values scale to max", () => {
  const pts = [0, 30_000, 70_000, 71_000].map((t, i) => ({ t, v: i + 1 }));
  const g = sparkGeometry(pts, 60_000, 600, 40)!;
  assert.ok(!g.line.includes("-710"), "the point 71 s old is gone");
  assert.equal(sparkGeometry(pts.slice(0, 1), 60_000, 600, 40), null);
  const flat = sparkGeometry(
    [
      { t: 0, v: 0 },
      { t: 1000, v: 0 },
      { t: 2000, v: 0 },
    ],
    60_000,
    100,
    20,
  )!;
  assert.equal(
    flat.max,
    1,
    "an all-zero series still scales against a floor of 1",
  );
});

test("spark: the slide is linear in time and never goes right", () => {
  assert.equal(slide(0, 0.01), -0);
  assert.equal(slide(1000, 0.01), -10);
  assert.equal(slide(-50, 0.01), -0);
});
