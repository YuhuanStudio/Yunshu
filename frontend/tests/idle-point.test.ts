import assert from "node:assert/strict";
import test from "node:test";
import { pointFromStatus } from "../src/series.ts";
import type { EngineStatus } from "../src/api.ts";

const status = (
  over: Partial<EngineStatus["requests"]>,
  live: number | null = null,
) =>
  ({
    throughput: { live_decode_tps: live },
    memory: {},
    requests: {
      active: 0,
      queued: 0,
      prefill: 0,
      decode: 0,
      items: [],
      ...over,
    },
  }) as unknown as EngineStatus;

test("nothing decoding is 0 tok/s, a number is a number, an unmeasured busy state stays unknown", () => {
  assert.equal(pointFromStatus(1, status({})).decode, 0);
  assert.equal(pointFromStatus(1, status({})).prefill, 0);
  assert.equal(pointFromStatus(1, status({ decode: 1 }, 48)).decode, 48);
  assert.equal(
    pointFromStatus(1, status({ decode: 1 }, null)).decode,
    null,
    "decoding but not measured yet: not a made-up 0",
  );
  assert.equal(pointFromStatus(1, status({ prefill: 1 })).prefill, null);
});
