import assert from "node:assert/strict";
import { test } from "node:test";
import { decodeSparkline, modelKind } from "../src/status-island.ts";
import type { SeriesPoint } from "../src/series.ts";

const pt = (at: number, decode: number | null, gap = false): SeriesPoint =>
  ({
    at,
    decode,
    prefill: null,
    active: null,
    queued: null,
    prefillRequests: null,
    decodeRequests: null,
    memActive: null,
    memCache: null,
    ...(gap ? { gap: true as const } : {}),
  }) as SeriesPoint;

test("sparkline keeps the last 60 s and treats null decode as 0 tok/s", () => {
  const s = Array.from({ length: 40 }, (_, i) =>
    pt(i * 3000, i % 2 ? 50 : null),
  );
  const out = decodeSparkline(s);
  assert.equal(out.length, 21);
  assert.equal(out[1], 0);
  assert.equal(out.at(-1), 50);
});

test("sparkline is empty below five samples", () => {
  assert.deepEqual(
    decodeSparkline([pt(0, 10), pt(3000, 11), pt(6000, 12)]),
    [],
  );
});

test("sparkline never bridges an outage marker", () => {
  const s = [
    ...Array.from({ length: 8 }, (_, i) => pt(i * 3000, 40)),
    pt(24000, null, true),
    pt(27000, 30),
    pt(30000, 30),
  ];
  assert.deepEqual(decodeSparkline(s), []);
});

test("model kind", () => {
  assert.equal(modelKind({ type: "VLMEngine" } as never), "VLM");
  assert.equal(modelKind({ type: "LLM" } as never), "LLM");
  assert.equal(modelKind(undefined), null);
});
