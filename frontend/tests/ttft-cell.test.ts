import assert from "node:assert/strict";
import test from "node:test";
import { totalCell, ttftCell } from "../src/request-insight.ts";

// The console showed a running request's elapsed time in the TTFT column, so "TTFT" kept
// growing through decode. TTFT freezes at the first token.
test("a decoding row keeps its first-token time as it runs", () => {
  const at = (elapsed: number) => ({
    phase: "decode",
    elapsed_s: elapsed,
    offsets_ms: { first_token: 1234 },
  });
  assert.equal(ttftCell(at(2)), 1234);
  assert.equal(ttftCell(at(60)), 1234);
  assert.equal(ttftCell(at(900)), 1234);
});

test("before the first token there is no TTFT, not the time waited", () => {
  assert.equal(ttftCell({ phase: "queued", elapsed_s: 30 }), null);
  assert.equal(ttftCell({ phase: "prefill", elapsed_s: 8, offsets_ms: { admit: 100 } }), null);
});

test("the total column carries the running time until the request finishes", () => {
  assert.deepEqual(totalCell({ phase: "decode", elapsed_s: 42 }), { ms: 42000, running: true });
  assert.deepEqual(
    totalCell({ phase: "complete", offsets_ms: { done: 5000 } }),
    { ms: 5000, running: false },
  );
});

import { phaseDurations } from "../src/request-insight.ts";

test("a decoding row keeps what its queue and prefill took, even when a poll never saw the prefill", () => {
  assert.deepEqual(
    phaseDurations({
      prompt_tokens: 10000,
      cached_tokens: 9200,
      offsets_ms: { admit: 40, first_token: 460 },
    }),
    { queueMs: 40, prefillMs: 420, cachedPct: 92 },
  );
  assert.equal(phaseDurations({ offsets_ms: { admit: 40 } }), null);
  assert.equal(phaseDurations({}), null);
});
