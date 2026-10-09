import assert from "node:assert/strict";
import test from "node:test";
import {
  distribution,
  groupOf,
  histogram,
} from "../src/latency-distribution.ts";

const row = (over: Record<string, unknown>) => ({
  ttft_ms: 100,
  queue_wait_ms: 10,
  prompt_tokens: 1000,
  cached_tokens: 0,
  model: "m",
  outcome: "completed" as const,
  ...over,
});

test("warm means at least half the prompt was cached; unknown sizes are not placed", () => {
  assert.equal(groupOf(row({ cached_tokens: 500 }), "warmth"), "warm");
  assert.equal(groupOf(row({ cached_tokens: 499 }), "warmth"), "cold");
  assert.equal(groupOf(row({ prompt_tokens: 0 }), "warmth"), null);
  assert.equal(groupOf(row({ cached_tokens: undefined }), "warmth"), null);
  assert.equal(groupOf(row({ prompt_tokens: 999 }), "context"), "ctx0");
  assert.equal(groupOf(row({ prompt_tokens: 1000 }), "context"), "ctx1");
  assert.equal(groupOf(row({ prompt_tokens: 40_000 }), "context"), "ctx3");
});

test("percentiles need 20 samples per group; below that only n is shown", () => {
  const rows = [
    ...Array.from({ length: 25 }, (_, i) =>
      row({ ttft_ms: 100 + i, cached_tokens: 900 }),
    ),
    ...Array.from({ length: 5 }, () =>
      row({ ttft_ms: 9000, cached_tokens: 0 }),
    ),
  ];
  const { groups } = distribution(rows, "warmth", "ttft");
  const warm = groups.find((g) => g.id === "warm")!;
  const cold = groups.find((g) => g.id === "cold")!;
  assert.equal(warm.n, 25);
  assert.ok(warm.p50 != null && warm.p90 != null);
  assert.equal(cold.n, 5);
  assert.equal(cold.p50, null);
  assert.equal(cold.p90, null);
});

test("errors are left out of TTFT but queue wait counts; missing metrics are not zeros", () => {
  const rows = [
    row({ outcome: "error", ttft_ms: 5, queue_wait_ms: 7 }),
    row({ ttft_ms: null, queue_wait_ms: 3 }),
  ];
  assert.equal(distribution(rows, "all", "ttft").groups.length, 0);
  const q = distribution(rows, "all", "queue").groups[0];
  assert.equal(q.n, 2);
  assert.equal(histogram(rows, "ttft").length, 0);
});

test("histogram covers every sample exactly once", () => {
  const rows = [50, 100, 249, 250, 12_000].map((ttft_ms) => row({ ttft_ms }));
  const h = histogram(rows, "ttft");
  assert.equal(
    h.reduce((n, b) => n + b.count, 0),
    5,
  );
  assert.equal(h.find((b) => b.min === 100)!.count, 2);
  assert.equal(h.at(-1)!.count, 1);
});
