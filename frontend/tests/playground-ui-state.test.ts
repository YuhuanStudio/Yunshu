import assert from "node:assert/strict";
import test from "node:test";
import { thinkingOpen } from "../src/playground-ui-state.ts";

test("reasoning opens while it streams and folds when the answer starts", () => {
  assert.equal(thinkingOpen(true, false, null), true);
  assert.equal(thinkingOpen(true, true, null), false);
  assert.equal(thinkingOpen(false, true, null), false);
});

test("a user toggle always wins", () => {
  assert.equal(thinkingOpen(true, true, true), true);
  assert.equal(thinkingOpen(true, false, false), false);
  assert.equal(thinkingOpen(false, true, true), true);
});
