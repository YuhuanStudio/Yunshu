import assert from "node:assert/strict";
import { test } from "node:test";
import { parseUsage } from "../src/stream.ts";

test("usage chunk carries x_yunshu timing, cache and speculative stats", () => {
  const usage = parseUsage({
    usage: {
      prompt_tokens: 1024,
      completion_tokens: 300,
      prompt_tokens_details: { cached_tokens: 512 },
    },
    x_yunshu: {
      ttft_ms: 180.4,
      speculative: {
        mode: "dflash",
        drafted: 900,
        accepted: 540,
        acceptance_rate: 0.6,
        rounds: 75,
      },
    },
  });
  assert.deepEqual(usage, {
    promptTokens: 1024,
    completionTokens: 300,
    cachedTokens: 512,
    ttftMs: 180.4,
    spec: { mode: "dflash", rounds: 75, acceptanceRate: 0.6 },
  });
});

test("no spec block when the engine did not draft", () => {
  const usage = parseUsage({
    usage: { completion_tokens: 3 },
    x_yunshu: { ttft_ms: 9 },
  });
  assert.equal(usage?.spec, undefined);
});

import { parseLogprobs } from "../src/stream.ts";
test("logprobs: valid entries kept with runner-ups, malformed ones dropped, never 0", () => {
  const tokens = parseLogprobs({
    content: [
      { token: "a", logprob: -0.1, top_logprobs: [{ token: "a", logprob: -0.1 }, { token: "b", logprob: -2.3 }, { token: 5 }] },
      { token: "b" },
      { logprob: -1 },
      null,
      { token: "c", logprob: Number.NaN },
    ],
  });
  assert.equal(tokens.length, 1);
  assert.equal(tokens[0].alternatives?.length, 2);
  assert.deepEqual(parseLogprobs(null), []);
  assert.deepEqual(parseLogprobs({ content: "x" }), []);
});
