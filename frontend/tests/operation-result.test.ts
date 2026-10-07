import assert from "node:assert/strict";
import test from "node:test";
import { operationResult } from "../src/operation-result.ts";
test("HTTP 200 warmup warnings are partial outcomes, not successful generation", () => {
  const result = operationResult("warmup:model", {
    generated: false,
    warning: "warm-up generation failed: HTTP 503",
  });
  assert.equal(result.error, true);
  assert.match(result.text, /HTTP 503/);
});
test("non-generative warmup and accepted cancellation preserve backend meaning", () => {
  assert.match(
    operationResult("warmup:model", { generated: false }).text,
    /未執行文字預熱/,
  );
  assert.match(operationResult("cancel:request", {}).text, /已送出取消/);
  assert.equal(operationResult("copy:model", undefined).error, false);
});
