import assert from "node:assert/strict";
import test from "node:test";
import { prefillSplit } from "../src/prefill-split.ts";

test("a cache hit is its own part: 8,000 cached of 10,000 starts the computed bar at 0, not at 80%", () => {
  const s = prefillSplit({ prompt_tokens: 10000, cached_tokens: 8000, processed_tokens: 8000, percent: 0 });
  assert.ok(s);
  assert.equal(s.cached, 8000);
  assert.equal(s.computed, 2000);
  assert.equal(s.done, 0);
  assert.equal(s.percentOfComputed, 0);
});

test("the engine's percent is a share of the computed part", () => {
  const s = prefillSplit({ prompt_tokens: 10000, cached_tokens: 8000, processed_tokens: 9000, percent: 50 });
  assert.equal(s?.done, 1000);
  assert.equal(s?.percentOfComputed, 50);
});

test("without percent, processed tokens (which include the cache) give the same answer", () => {
  const s = prefillSplit({ prompt_tokens: 10000, cached_tokens: 8000, processed_tokens: 9500 });
  assert.equal(s?.done, 1500);
  assert.equal(s?.percentOfComputed, 75);
});

test("no cache: plain progress; unknown stays null, never 0", () => {
  assert.equal(prefillSplit({ prompt_tokens: 1000, processed_tokens: 250 })?.percentOfComputed, 25);
  assert.equal(prefillSplit({ prompt_tokens: 1000 }), null);
  assert.equal(prefillSplit({}), null);
});

test("fully cached prompt reads as complete; an oversized cache hit is clamped", () => {
  assert.equal(prefillSplit({ prompt_tokens: 500, cached_tokens: 500, processed_tokens: 500 })?.percentOfComputed, 100);
  assert.equal(prefillSplit({ prompt_tokens: 500, cached_tokens: 900, processed_tokens: 500 })?.cached, 500);
});

test("only a percentage and no prompt size: shown as the engine said it", () => {
  const s = prefillSplit({ percent: 42 });
  assert.equal(s?.percentOfComputed, 42);
  assert.equal(s?.prompt, 0);
});
