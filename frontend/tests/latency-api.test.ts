import assert from "node:assert/strict";
import test from "node:test";
import { parseLatency } from "../src/latency-api.ts";

const sample = {
  milestones_ms: {
    gateway_admit: 1,
    template_start: 2,
    template_end: 12,
    prefill_start: 14,
    prefill_end: 214,
    first_decode: 230,
    sse_first_flush: 231,
  },
  durations_ms: {
    model_lease: null,
    gateway_admit: 1,
    engine_queue: null,
    template_tokenize: 10,
    apc_lookup_restore: null,
    prefill: 200,
    first_decode: 16,
    sse_first_flush: 1,
  },
};

test("unobserved stages stay null, never 0; spans come only from stages with marks", () => {
  const l = parseLatency(sample)!;
  assert.equal(l.durations.model_lease, null);
  assert.equal(l.durations.engine_queue, null);
  assert.equal(l.durations.prefill, 200);
  assert.deepEqual(
    l.spans.map((s) => s.id),
    [
      "gateway_admit",
      "template_tokenize",
      "prefill",
      "first_decode",
      "sse_first_flush",
    ],
  );
  assert.deepEqual(
    l.spans.find((s) => s.id === "prefill"),
    {
      id: "prefill",
      start: 14,
      end: 214,
    },
  );
});

test("garbage and negative values are dropped", () => {
  assert.equal(parseLatency(null), null);
  assert.equal(parseLatency({}), null);
  const l = parseLatency({ durations_ms: { prefill: -5, first_decode: "x" } })!;
  assert.equal(l.durations.prefill, null);
  assert.equal(l.durations.first_decode, null);
});

import { parseEnergy } from "../src/latency-api.ts";

test("energy receipts: a phase without coverage stays unknown with null joules, never 0", () => {
  const e = parseEnergy({
    schema: "yunshu.energy.v1",
    prefill: {
      state: "unknown",
      reason: "no coverage",
      joules: null,
      joules_per_token: null,
    },
    decode: {
      state: "estimated",
      joules: 32,
      joules_per_token: 0.4,
      gpu_watts_mean: 30,
      coverage_ratio: 1,
      extrapolated_s: 0.25,
    },
  })!;
  assert.equal(e.state, "estimated");
  assert.equal(e.prefill!.joules, null);
  assert.equal(e.prefill!.reason, "no coverage");
  assert.equal(e.decode!.joulesPerToken, 0.4);
  const off = parseEnergy({
    state: "unknown",
    reason: "telemetry disabled or engine timing unavailable",
  })!;
  assert.equal(off.state, "unknown");
  assert.equal(off.prefill, null);
  assert.equal(parseEnergy(null), null);
});
