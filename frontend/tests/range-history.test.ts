import assert from "node:assert/strict";
import test from "node:test";
import {
  LIVE_RANGE_MAX_S,
  RANGES,
  rangeSeconds,
  stepFor,
} from "../src/useRangeHistory.ts";
import { fetchStatus } from "../src/api.ts";

test("the ranges are 15 m, 1 h, 6 h, 24 h, 7 d and 30 d; only the first two come from the live series", () => {
  assert.deepEqual(
    RANGES.map((r) => r.id),
    ["15m", "1h", "6h", "24h", "7d", "30d"],
  );
  assert.deepEqual(
    RANGES.map((r) => r.seconds),
    [900, 3600, 21600, 86400, 604800, 2592000],
  );
  assert.equal(LIVE_RANGE_MAX_S, 3600);
  assert.equal(rangeSeconds("nope"), 900);
});

test("a range asks the server for about 600 points, never finer than a second", () => {
  assert.equal(stepFor(900), 1.5);
  assert.equal(stepFor(3600 * 6), 36);
  assert.equal(stepFor(30 * 86400), 4320);
  assert.equal(stepFor(60), 1);
});

const statusBody = {
  object: "yunshu.status",
  version: "t",
  state: "running",
  uptime_s: 1,
  load_error: null,
  models: [],
  memory: {},
  requests: { active: 0, queued: 0, prefill: 0, decode: 0, items: [] },
  last: null,
  throughput: {
    window_s: 60,
    requests: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    live_decode_tps: null,
    mean_prefill_tps: null,
    mean_decode_tps: null,
  },
};

async function withFetch(headers: Record<string, string>) {
  const original = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response(JSON.stringify(statusBody), {
      status: 200,
      headers: { "content-type": "application/json", ...headers },
    })) as typeof fetch;
  try {
    return await fetchStatus({ baseUrl: "http://127.0.0.1:8100", token: "" });
  } finally {
    globalThis.fetch = original;
  }
}

test("a status answer stamped by the console process says so; an engine's own answer does not", async () => {
  assert.equal(
    (await withFetch({ "x-yunshu-console": "1" })).console_process,
    true,
  );
  assert.equal((await withFetch({})).console_process, undefined);
});
