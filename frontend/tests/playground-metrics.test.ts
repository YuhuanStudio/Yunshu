import assert from "node:assert/strict";
import test from "node:test";
import {
  compareOutputs,
  firstDivergence,
  runStats,
} from "../src/playground-metrics.ts";

test("runStats uses usage tokens and excludes TTFT from decode rate", () => {
  const s = runStats(
    {
      start: 0,
      firstAt: 500,
      endAt: 1500,
      chunks: 5,
      usage: { completionTokens: 41, cachedTokens: 512, promptTokens: 1024 },
    },
    9999,
  );
  assert.equal(s.tokens, 41);
  assert.equal(s.estimated, false);
  assert.equal(s.ttftMs, 500);
  assert.equal(s.tokensPerSecond, 40);
  assert.equal(s.latencyMs, 1500);
  assert.equal(s.cachedTokens, 512);
});

test("runStats estimates from chunks while streaming", () => {
  const s = runStats({ start: 100, firstAt: 200, chunks: 3 }, 400);
  assert.equal(s.estimated, true);
  assert.equal(s.tokens, 3);
  assert.equal(s.latencyMs, 300);
  assert.equal(s.tokensPerSecond, 10);
  assert.equal(runStats({ start: 0, chunks: 0 }, 50).ttftMs, undefined);
});

test("firstDivergence and compareOutputs are exact", () => {
  assert.equal(firstDivergence("abc", "abc"), null);
  assert.equal(firstDivergence("abcd", "abxd"), 2);
  assert.equal(firstDivergence("abc", "abcd"), 3);
  assert.deepEqual(
    compareOutputs(
      { text: "x", temperature: 0 },
      { text: "x", temperature: 0 },
    ),
    { kind: "identical", greedy: true },
  );
  assert.deepEqual(
    compareOutputs(
      { text: "x", temperature: 0 },
      { text: "x", temperature: 0.7 },
    ),
    { kind: "identical", greedy: false },
  );
  assert.deepEqual(
    compareOutputs(
      { text: "x ", temperature: 0 },
      { text: "x", temperature: 0 },
    ),
    { kind: "diverged", offset: 1 },
  );
});
