import assert from "node:assert/strict";
import test from "node:test";
import { WidthRatchet } from "../src/motion/hug-core.ts";
import { READOUT_MS, Readout } from "../src/motion/readout-core.ts";
import {
  analyze,
  widthEvents,
  widthSeries,
} from "../scripts/stability-analyze.mjs";

test("a box grows at once and gives room back only after the hold", () => {
  const r = new WidthRatchet(3000);
  assert.equal(r.observe(80, 0), 80, "first measure is taken as is");
  assert.equal(r.observe(87, 250), 87, "a new digit grows the box immediately");
  assert.equal(r.observe(80, 500), 87, "a narrower sample waits");
  assert.equal(
    r.observe(87, 750),
    87,
    "the wider value is back: hold restarts",
  );
  assert.equal(r.observe(80, 1000), 87);
  assert.equal(r.dueIn(1000), 3000);
  assert.equal(r.observe(80, 3900), 87, "not yet: hold counts from 1000");
  assert.equal(r.observe(80, 4000), 80, "held narrow for 3 s: shrinks");
  assert.equal(r.dueIn(4000), null);
});

test("a flickering digit count never resizes after the first growth", () => {
  const r = new WidthRatchet(3000);
  const seen = new Set<number>();
  for (let i = 0; i < 80; i++) seen.add(r.observe(i % 2 ? 87 : 80, i * 250));
  assert.equal(seen.size, 2, "one growth, then no change in 20 s");
});

test("shrinks to the widest the content was during the hold, not to the last sample", () => {
  const r = new WidthRatchet(1000);
  r.observe(100, 0);
  r.observe(60, 100);
  r.observe(70, 600);
  r.observe(55, 800);
  assert.equal(r.observe(55, 1200), 70);
  assert.equal(r.settle(5000), 70);
});

test("a shrink that is undone soon doubles the hold; a lasting one restores it", () => {
  const r = new WidthRatchet(1000);
  r.observe(100, 0);
  r.observe(60, 10);
  assert.equal(r.observe(60, 1010), 60, "first shrink after 1 s");
  assert.equal(r.observe(100, 1500), 100, "regretted: needed the room again");
  r.observe(60, 1600);
  assert.equal(r.dueIn(1600), 2000, "hold is now 2 s");
  assert.equal(r.observe(60, 3500), 100, "still holding at 1.9 s");
  assert.equal(r.observe(60, 3600), 60);
  r.observe(100, 20000);
  r.observe(60, 20010);
  assert.equal(
    r.dueIn(20010),
    1000,
    "an undisturbed shrink restores the base hold",
  );
});

test("a pending shrink is applied by settle when no sample arrives", () => {
  const r = new WidthRatchet(1000);
  r.observe(100, 0);
  r.observe(40, 10);
  assert.equal(r.dueIn(500), 510);
  assert.equal(r.settle(1010), 40);
});

test("a readout draws at most every READOUT_MS and the newest waiting sample wins", () => {
  const ro = new Readout<number>();
  assert.equal(ro.offer(1, 0), 1);
  assert.equal(ro.offer(2, 100), null);
  assert.equal(ro.offer(3, 200), null);
  assert.equal(ro.dueIn(200), READOUT_MS - 200);
  assert.equal(ro.flush(200), null, "too early");
  assert.equal(
    ro.flush(READOUT_MS),
    3,
    "the newest sample is drawn, 2 never is",
  );
  assert.equal(ro.dueIn(READOUT_MS), null);
  assert.equal(
    ro.offer(4, READOUT_MS + 10),
    null,
    "interval restarts at the draw",
  );
});

test("a jump draws at once and drops the waiting sample", () => {
  const ro = new Readout<number | null>();
  ro.offer(1, 0);
  ro.offer(2, 50);
  assert.equal(ro.jump(null, 60), null);
  assert.equal(ro.flush(1000), null);
  assert.equal(ro.offer(5, 60 + READOUT_MS), 5);
});

test("at 4 Hz input a readout passes every sample; at 60 Hz it passes about 4 a second", () => {
  const draws = (hz: number) => {
    const ro = new Readout<number>();
    let n = 0;
    for (let t = 0; t < 2000; t += 1000 / hz) {
      if (ro.offer(t, t) != null) n++;
      if (ro.flush(t) != null) n++;
    }
    return n / 2;
  };
  assert.ok(draws(4) >= 3.5, `4 Hz input: ${draws(4)}/s`);
  assert.ok(draws(60) <= 4.3, `60 Hz input: ${draws(60)}/s`);
});

test("widthEvents counts an eased step once and spots A -> B -> A", () => {
  const eased = [100, 100, 103, 108, 112, 112, 112].map((w, i) => ({
    t: i * 50,
    w,
  }));
  const e = widthEvents(eased);
  assert.equal(e.steps, 1);
  assert.equal(e.flips, 0);
  const flip = [80, 80, 87, 87, 80, 80].map((w, i) => ({ t: i * 250, w }));
  assert.equal(widthEvents(flip).flips, 1);
  const spaced = [80, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 80].map(
    (w, i) => ({ t: i * 500, w }),
  );
  assert.equal(
    widthEvents(spaced).flips,
    0,
    "going back after 3 s is a settled change",
  );
});

test("analyze separates a resized element from one that was only shoved", () => {
  const frames = [
    {
      t: 0,
      scrollTop: 0,
      boxes: { 1: [0, 0, 80, 20], 2: [88, 0, 50, 20] },
      dead: {},
    },
    {
      t: 50,
      scrollTop: 0,
      boxes: { 1: [0, 0, 87, 20], 2: [95, 0, 50, 20] },
      dead: { 1: 4 },
    },
    {
      t: 100,
      scrollTop: 0,
      boxes: { 1: [0, 0, 87, 20], 2: [95, 0, 50, 20] },
      dead: {},
    },
  ];
  const a = analyze(frames, { 1: "pill", 2: "next" });
  assert.equal(a.steps, 1);
  assert.equal(a.shoves, 1);
  assert.deepEqual(a.shoved, [["next", 1]]);
  assert.equal(a.dead[0].px, 4);
  assert.deepEqual(
    widthSeries(frames, "1").map((s) => s.w),
    [80, 87, 87],
  );
  const scrolled = [
    { t: 0, scrollTop: 0, boxes: { 1: [0, 100, 10, 10] }, dead: {} },
    { t: 50, scrollTop: 30, boxes: { 1: [0, 70, 10, 10] }, dead: {} },
  ];
  assert.equal(
    analyze(scrolled, { 1: "x" }).moved,
    0,
    "scroll is not movement",
  );
});
