import assert from "node:assert/strict";
import test from "node:test";
import { createDeltaBatcher } from "../src/delta-batch.ts";

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
