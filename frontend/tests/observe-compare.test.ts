import assert from "node:assert/strict";
import test from "node:test";
import {
  anyDiffers,
  compareRequest,
  parseInFlight,
} from "../src/observe-compare.ts";

const ui = {
  id: "r1",
  phase: "decode",
  completion_tokens: 40,
  offsets_ms: { admit: 5, first_token: 900 },
};

test("parseInFlight keeps only rows with a request id", () => {
  assert.deepEqual(parseInFlight(null), []);
  assert.deepEqual(parseInFlight({ in_flight: "x" }), []);
  assert.equal(
    parseInFlight({ in_flight: [{ request_id: "a" }, { x: 1 }, 3] }).length,
    1,
  );
});

test("matching engine numbers produce no difference", () => {
  const cells = compareRequest(ui, {
    request_id: "r1",
    has_stats: true,
    phase: "decode",
    generated: 44,
    t_admit_ms: 5,
    t_first_ms: 901,
  });
  assert.equal(anyDiffers(cells), false);
});

test("a phase or timestamp the engine contradicts is flagged", () => {
  const cells = compareRequest(ui, {
    request_id: "r1",
    has_stats: true,
    phase: "prefill",
    generated: 0,
    t_admit_ms: 5,
    t_first_ms: null,
  });
  const by = Object.fromEntries(cells.map((c) => [c.key, c.state]));
  assert.equal(by.phase, "differs");
  assert.equal(by.tokens, "differs");
  assert.equal(by.first, "missing");
  assert.equal(anyDiffers(cells), true);
});

test("a request /debug does not know is missing, not a difference", () => {
  const cells = compareRequest(ui, undefined);
  assert.ok(cells.every((c) => c.state === "missing"));
});

test("a token gap the sampling skew explains is not a mismatch", () => {
  const fast = { ...ui, completion_tokens: 595, tokens_per_second: 105 };
  const dbg = {
    request_id: "r1",
    has_stats: true,
    phase: "decode",
    generated: 418,
  };
  assert.equal(anyDiffers(compareRequest(fast, dbg, 0)), true);
  assert.equal(anyDiffers(compareRequest(fast, dbg, 1)), true); // 177 > 16 + 1 * 105
  assert.equal(anyDiffers(compareRequest(fast, dbg, 2)), false); // 177 <= 16 + 2 * 105
});

test("admit and prefill start are compared each with its own engine field", () => {
  const row = {
    ...ui,
    offsets_ms: { admit: 30, prefill_start: 3245, first_token: 3300 },
  } as never;
  const dbg = {
    request_id: "r1",
    phase: "decode",
    generated: 40,
    t_admit_ms: 30.1,
    t_prefill_start_ms: 3245.2,
    t_first_ms: 3300,
  };
  const cells = compareRequest(row, dbg);
  assert.equal(cells.find((c) => c.key === "admit")?.state, "match");
  assert.equal(cells.find((c) => c.key === "prefillStart")?.state, "match");
  assert.equal(anyDiffers(cells), false);
});
