import { test } from "node:test";
import assert from "node:assert/strict";
import { BurstGovernor } from "../src/motion/burst-core.ts";

test("a calm screen animates", () => {
  const g = new BurstGovernor(2000, 2);
  assert.equal(g.permit(0), true);
  assert.equal(g.permit(500), true);
  assert.equal(g.permit(5000), true);
});

test("a burst is refused after the first few and stays refused while it keeps coming", () => {
  const g = new BurstGovernor(2000, 2);
  let granted = 0;
  // 20 updates a second for 30 s
  for (let t = 0; t < 30_000; t += 50) if (g.permit(t)) granted++;
  assert.equal(granted, 2);
});

test("a slow trickle (one per window) always animates", () => {
  const g = new BurstGovernor(2000, 2);
  for (let t = 0; t < 60_000; t += 1000) assert.equal(g.permit(t), true);
});

test("it animates again after the storm has been quiet for a window", () => {
  const g = new BurstGovernor(2000, 2);
  for (let t = 0; t < 5000; t += 100) g.permit(t);
  assert.equal(g.permit(5100), false);
  assert.equal(g.permit(7200), true);
});

test("animations started per second stay under a low bound for any rate", () => {
  for (const hz of [4, 10, 30]) {
    const g = new BurstGovernor();
    const starts: number[] = [];
    for (let t = 0; t < 60_000; t += 1000 / hz) if (g.permit(t)) starts.push(t);
    const busiest = Math.max(
      ...starts.map((s) => starts.filter((x) => x >= s && x < s + 1000).length),
    );
    assert.ok(busiest <= 2, `${hz} Hz: ${busiest} in a second`);
    assert.ok(starts.length <= 2, `${hz} Hz: ${starts.length} in a minute`);
  }
});
