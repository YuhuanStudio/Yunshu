import assert from "node:assert/strict";
import test from "node:test";
import { parseLifecycle, requestHitRate } from "../src/cache-lifecycle.ts";

const raw = {
  caches: [
    {
      model_id: "m",
      apc: {
        entries: 12,
        resident_bytes: 1000,
        warm_bytes: 200,
        warm_ratio: 2.5,
        disk_bytes: 5000,
        lookups_hit: 30,
        lookups_miss: 10,
        matched_tokens: 90000,
        memory_evictions: 0,
        memory_skips: 3,
        storage_tiers: [
          { cost_rejected: 2, invalidated: 1 },
          { cost_rejected: 1 },
        ],
      },
    },
    { model_id: "err", error: "unavailable" },
  ],
};

test("lifecycle keeps reported zeros, sums per-tier counters, and skips models without an apc block", () => {
  const [m, ...rest] = parseLifecycle(raw);
  assert.equal(rest.length, 0);
  const get = (id: string) => m.counters.find((c) => c.id === id)?.value;
  assert.equal(get("memory_evictions"), 0);
  assert.equal(get("memory_skips"), 3);
  assert.equal(get("cost_rejected"), 3);
  assert.equal(get("invalidated"), 1);
  assert.equal(get("warm_dropped"), undefined);
  assert.equal(get("matched_tokens"), 90000);
  assert.equal(m.bytes.warmRatio, 2.5);
});

test("request hit rate and cached tokens are separate quantities", () => {
  const [m] = parseLifecycle(raw);
  assert.equal(requestHitRate(m), 0.75);
  assert.equal(parseLifecycle({}).length, 0);
});
