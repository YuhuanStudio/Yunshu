import assert from "node:assert/strict";
import test from "node:test";
import { parseLatency, parseRecentLatency } from "../src/latency-api.ts";

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

test("an engine whose rows carry no latency field is unsupported; one with the field is ok", () => {
  assert.equal(
    parseRecentLatency({ data: [{ request_id: "a" }] }).kind,
    "unsupported",
  );
  const ok = parseRecentLatency({
    data: [
      { request_id: "a", latency: sample },
      { request_id: "b", latency: null },
    ],
  });
  assert.equal(ok.kind, "ok");
  if (ok.kind === "ok") {
    assert.ok(ok.byId.has("a"));
    assert.ok(!ok.byId.has("b"));
  }
  assert.equal(parseRecentLatency({ data: [] }).kind, "ok");
  assert.equal(parseRecentLatency("x").kind, "unsupported");
});
