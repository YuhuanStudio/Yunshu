import assert from "node:assert/strict";
import test from "node:test";
import { frameBudget } from "../browser/support/frame-budget.ts";

test("a single host scheduling pause is distinct from sustained application jank", () => {
  const idle = Array(150).fill(16.7);
  const app = [...idle];
  app[70] = 130;
  const once = frameBudget(app, idle);
  assert.ok(once.p95 < once.p95Limit && once.stalls <= once.stallLimit);
  app[80] = 140;
  assert.ok(frameBudget(app, idle).stalls > once.stallLimit);
  const slow = frameBudget(Array(150).fill(50), idle);
  assert.ok(slow.p95 > slow.p95Limit);
});

test("uses runner cadence and rejects incomplete output", () => {
  assert.throws(() => frameBudget([], []), /incomplete/);
  const result = frameBudget(Array(90).fill(33.3), Array(90).fill(33.3));
  assert.ok(result.p95 < result.p95Limit);
});
