import { test } from "node:test";
import assert from "node:assert/strict";
import { summarize } from "../scripts/anim-count.mjs";

test("events of one kind within 150 ms are one gesture", () => {
  const events = [0, 10, 20, 30].map((t) => ({
    t,
    name: "transition:transform li",
  }));
  const s = summarize(events, 10);
  assert.equal(s.total, 4);
  assert.equal(s.gestures, 1);
  assert.equal(s.peak, 1);
  assert.equal(s.perSecond, 0.1);
});

test("peak is the most gestures in any one second", () => {
  const events = [0, 400, 800, 5000].map((t) => ({
    t,
    name: "css:fresh-row-in tr",
  }));
  const s = summarize(events, 10);
  assert.equal(s.gestures, 4);
  assert.equal(s.peak, 3);
  assert.deepEqual(s.byName, [["css:fresh-row-in tr", 4]]);
});
