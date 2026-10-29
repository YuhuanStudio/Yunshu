import { test } from "node:test";
import assert from "node:assert/strict";
import { RowPacer } from "../src/motion/coalesce-core.ts";

test("the first list and a calm arrival are drawn at once", () => {
  const p = new RowPacer(1500);
  assert.deepEqual(p.offer(["a", "b"], 0), ["a", "b"]);
  assert.deepEqual(p.offer(["c", "a", "b"], 5000), ["c", "a", "b"]);
});

test("arrivals inside the interval wait and then show together in one step", () => {
  const p = new RowPacer(1500);
  p.offer(["a"], 0);
  p.offer(["b", "a"], 5000); // calm: drawn
  assert.deepEqual(p.offer(["c", "b", "a"], 5200), ["b", "a"]);
  assert.deepEqual(p.offer(["d", "c", "b", "a"], 5600), ["b", "a"]);
  assert.equal(p.dueIn(["d", "c", "b", "a"], 5600), 900);
  assert.deepEqual(p.offer(["d", "c", "b", "a"], 6500), ["d", "c", "b", "a"]);
  assert.equal(p.dueIn(["d", "c", "b", "a"], 6500), null);
});

test("a storm of 10 arrivals a second changes the table at most once per interval", () => {
  const p = new RowPacer(1500);
  let ids: string[] = [];
  const shifts: number[] = [];
  let shown: readonly string[] = [];
  for (let t = 0; t < 30_000; t += 100) {
    ids = [`r${t}`, ...ids].slice(0, 40);
    const next = p.offer(ids, t);
    if (next !== shown) shifts.push(t);
    shown = next;
  }
  const gaps = shifts.slice(1).map((t, i) => t - shifts[i]);
  assert.ok(
    gaps.every((g) => g >= 1500),
    `gaps ${gaps}`,
  );
  assert.ok(shifts.length <= 21, `${shifts.length} shifts in 30 s`);
});

test("another filter or sort shows its list at once", () => {
  const p = new RowPacer(1500);
  p.offer(["a", "b"], 0);
  assert.deepEqual(p.offer(["x", "y"], 100, true), ["x", "y"]);
  // and the interval restarts from there
  assert.deepEqual(p.offer(["z", "x", "y"], 200), ["x", "y"]);
});
