import assert from "node:assert/strict";
import test from "node:test";
import { liveSlots } from "../src/live-slots.ts";

test("arrivals, departures and phase changes cannot move a surviving request", () => {
  const first = liveSlots([], [{ id: "a" }, { id: "b" }, { id: "c" }]);
  const updated = { id: "b", phase: "decode" };
  const next = liveSlots(first, [{ id: "d" }, updated, { id: "c" }]);
  assert.equal(next.length, 3);
  assert.equal(next[0]?.id, "d");
  assert.equal(next[1], updated);
  assert.equal(next[2]?.id, "c");
  assert.deepEqual(liveSlots(next, []), Array(3).fill(null));
});

test("bursts beyond the initial lanes keep every request without shrinking on completion", () => {
  const rows = Array.from({ length: 25 }, (_, i) => ({ id: String(i) }));
  const slots = liveSlots([], rows);
  assert.equal(slots.length, 25);
  assert.equal(liveSlots(slots, [rows[24]])[24], rows[24]);
  assert.equal(liveSlots(slots, [rows[24]]).length, 25);
});
