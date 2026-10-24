import assert from "node:assert/strict";
import test from "node:test";
import { parseServerHistory } from "../src/history-api.ts";

const base = {
  series: { t: [1, 2, 3], decode_tps: [1, 2, 3], requests_active: [0, 0, 0] },
};

test("recorded restarts, switches and engine-away events become boundary marks (epoch ms)", () => {
  const h = parseServerHistory({
    ...base,
    events: [
      { t: 10, kind: "engine_unreachable", detail: { after_up: true } },
      { t: 11, kind: "engine_unreachable", detail: { after_up: false } },
      { t: 20, kind: "engine_restarted", detail: { pid: 1 } },
      { t: 30, kind: "engine_switched", detail: { from: "a", to: "b" } },
      { t: 31, kind: "model_loaded", detail: { model: "m" } },
    ],
  });
  assert.deepEqual(h?.restarts, [
    { at: 10_000, kind: "away" },
    { at: 20_000, kind: "restarted" },
    { at: 30_000, kind: "switched" },
  ]);
});

test("no events means no marks", () => {
  assert.deepEqual(parseServerHistory(base)?.restarts, []);
});
