import assert from "node:assert/strict";
import test from "node:test";
import { summarizeSpeculation } from "../src/speculation.ts";

test("acceptance is token weighted over requests that reported both counters", () => {
  const s = summarizeSpeculation([
    { speculative: { mode: "mtp", drafted: 100, accepted: 80, rounds: 20 } },
    { speculative: { mode: "mtp", drafted: 10, accepted: 0, rounds: 2 } },
    { speculative: { mode: "mtp", acceptance_rate: 0.9 } },
    { speculative: null },
    {},
  ]);
  assert.equal(s.total, 5);
  assert.equal(s.engaged, 3);
  assert.equal(s.plain, 2);
  assert.equal(s.unattributed, 1);
  const m = s.modes[0];
  assert.equal(m.mode, "mtp");
  assert.equal(m.requests, 3);
  assert.equal(m.counted, 2);
  assert.equal(m.drafted, 110);
  assert.equal(m.accepted, 80);
  assert.ok(Math.abs(m.acceptance! - 80 / 110) < 1e-9);
  assert.equal(m.rounds, 22);
});

test("no drafted tokens means no acceptance figure, never zero", () => {
  const s = summarizeSpeculation([{ speculative: { mode: "ngram" } }]);
  assert.equal(s.modes[0].acceptance, null);
  assert.equal(s.unattributed, 1);
});
