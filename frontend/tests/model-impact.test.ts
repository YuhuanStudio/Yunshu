import assert from "node:assert/strict";
import test from "node:test";
import type { EngineStatus } from "../src/api.ts";
import { unloadImpact } from "../src/model-impact.ts";

const status = (items: unknown[], models: string[]) =>
  ({
    models: models.map((id) => ({ id, loaded: true, loading: false })),
    requests: { active: items.length, queued: 0, prefill: 0, decode: 0, items },
  }) as unknown as EngineStatus;

test("rows of the model are listed, other models' rows are not", () => {
  const s = status(
    [
      { request_id: "a", model: "m1", phase: "decode", elapsed_s: 1 },
      { request_id: "b", model: "m2", phase: "decode", elapsed_s: 1 },
    ],
    ["m1", "m2"],
  );
  const i = unloadImpact(s, "m1");
  assert.deepEqual(
    i.rows.map((r) => r.request_id),
    ["a"],
  );
  assert.equal(i.unattributed, 0);
});

test("a row with no model belongs to the only loaded model, else it is unattributed", () => {
  const items = [{ request_id: "a", phase: "prefill", elapsed_s: 1 }];
  assert.equal(unloadImpact(status(items, ["m1"]), "m1").rows.length, 1);
  const two = unloadImpact(status(items, ["m1", "m2"]), "m1");
  assert.equal(two.rows.length, 0);
  assert.equal(two.unattributed, 1);
});

test("no status, no impact; cancelled rows do not count", () => {
  assert.deepEqual(unloadImpact(null, "m1"), { rows: [], unattributed: 0 });
  const s = status(
    [
      {
        request_id: "a",
        model: "m1",
        phase: "decode",
        elapsed_s: 1,
        cancelled: true,
      },
    ],
    ["m1"],
  );
  assert.equal(unloadImpact(s, "m1").rows.length, 0);
});
