import assert from "node:assert/strict";
import test from "node:test";
import { parsePrepare } from "../src/admin-models-api.ts";

test("a running prepare job and an inapplicable model parse", () => {
  const running = parsePrepare({
    applicable: true,
    prepared: false,
    storage: "ssd",
    table_bytes: 3.2e10,
    job: { state: "running", bytes_done: 1e9, bytes_total: 3.2e10, eta_s: 90 },
  });
  assert.equal(running.job?.state, "running");
  assert.equal(running.job?.etaS, 90);
  assert.equal(running.storage, "ssd");
  assert.equal(parsePrepare({ applicable: false }).applicable, false);
  assert.equal(
    parsePrepare({ applicable: true, job: { state: "?" } }).job?.state,
    "failed",
  );
});
