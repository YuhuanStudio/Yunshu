import assert from "node:assert/strict";
import test from "node:test";
import {
  SHRINK_AFTER_MS,
  initialAxis,
  nextAxis,
  niceCeil,
  timeTicks,
} from "../src/motion/axis-core.ts";

test("nice maxima are 1, 2, 2.5, 5 x 10^k", () => {
  assert.deepEqual(
    [0, 0.4, 1, 1.1, 2.2, 3, 7, 18, 51, 95, 120, 260, 2600].map(niceCeil),
    [1, 1, 1, 2, 2.5, 5, 10, 20, 100, 100, 200, 500, 5000],
  );
});

test("the axis grows at once and never ends below the data", () => {
  let a = initialAxis(40);
  assert.equal(a.max, 50);
  a = nextAxis(a, 95, 1000);
  assert.equal(a.max, 100);
  a = nextAxis(a, 400, 2000);
  assert.ok(a.max >= 400);
});

test("it shrinks only after the data stays under 60 percent of the axis for a while", () => {
  let a = initialAxis(95); // 100
  a = nextAxis(a, 30, 0);
  assert.equal(a.max, 100, "not at once");
  a = nextAxis(a, 31, SHRINK_AFTER_MS - 1);
  assert.equal(a.max, 100, "not yet");
  a = nextAxis(a, 32, SHRINK_AFTER_MS + 1);
  assert.equal(a.max, 50, "now, to a nice value that still fits");
});

test("a value back above the threshold cancels the shrink; in between values never trigger it", () => {
  let a = initialAxis(95);
  a = nextAxis(a, 30, 0);
  a = nextAxis(a, 70, 4000); // above 60 %: the timer resets
  a = nextAxis(a, 30, 5000);
  a = nextAxis(a, 30, 5000 + SHRINK_AFTER_MS - 1);
  assert.equal(a.max, 100);
  const flat = nextAxis(initialAxis(60), 55, 1e9);
  assert.equal(flat.max, 100);
});

test("a flat idle zero keeps a floor of 1 and never shrinks below it", () => {
  let a = initialAxis(0);
  assert.equal(a.max, 1);
  a = nextAxis(a, 0, 1e6);
  assert.equal(a.max, 1);
});

test("time ticks are absolute multiples of a round step inside the window", () => {
  const ticks = timeTicks(1_000_000_000_000 - 300_000, 1_000_000_000_000, 4);
  assert.ok(ticks.length >= 3 && ticks.length <= 6);
  assert.ok(
    ticks.every(
      (t) =>
        t % 60_000 === 0 ||
        t % 30_000 === 0 ||
        t % 15_000 === 0 ||
        t % 10_000 === 0,
    ),
  );
  const later = timeTicks(1_000_000_002_500 - 300_000, 1_000_000_002_500, 4);
  assert.deepEqual(
    ticks.filter((t) => later.includes(t)),
    ticks.filter((t) => t >= 1_000_000_002_500 - 300_000),
    "the same ticks while the window slides",
  );
});
