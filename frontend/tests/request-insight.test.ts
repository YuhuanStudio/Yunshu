import assert from "node:assert/strict";
import test from "node:test";
import {
  diagnose,
  isSlow,
  percentile,
  slowRule,
  sortRows,
  SLOW_TOTAL_MS,
  SLOW_TTFT_MS,
} from "../src/request-insight.ts";

const base = {
  prompt_tokens: 45000,
  cached_tokens: 4000,
  completion_tokens: 100,
  decode_tps: 40,
};

test("a long cold prefill is named with its fresh token count", () => {
  const c = diagnose({
    ...base,
    ttft_ms: 41000,
    offsets_ms: { admit: 100, first_token: 41100, done: 43600 },
  });
  assert.equal(c.kind, "prefillMiss");
  if (c.kind === "prefillMiss") assert.equal(c.fresh, 41000);
});

test("queue wait wins when it dominates", () => {
  const c = diagnose({
    ...base,
    offsets_ms: { admit: 12000, first_token: 12600, done: 14000 },
  });
  assert.equal(c.kind, "queue");
});

test("a reload that is most of the prefill is a reload, not a miss", () => {
  const c = diagnose({
    ...base,
    cache: { tier: "ssd", reload_ms: 6000 },
    offsets_ms: { admit: 10, first_token: 8010, done: 9000 },
  });
  assert.equal(c.kind, "prefillReload");
});

test("slow decode: low acceptance first, then low tok/s, else just long output", () => {
  const o = { admit: 10, first_token: 300, done: 30300 };
  const spec = diagnose({
    ...base,
    speculative: { acceptance_rate: 0.18 },
    offsets_ms: o,
  });
  assert.equal(spec.kind, "decodeSpec");
  assert.equal(
    diagnose({ ...base, decode_tps: 9, offsets_ms: o }).kind,
    "decodeSlow",
  );
  assert.equal(diagnose({ ...base, offsets_ms: o }).kind, "decodeLong");
});

test("no timestamps is unknown, fast requests get no diagnosis, errors stay errors", () => {
  assert.equal(diagnose({ prompt_tokens: 5 }).kind, "unknown");
  assert.equal(
    diagnose({ offsets_ms: { admit: 1, first_token: 200, done: 900 } }).kind,
    "fast",
  );
  assert.equal(diagnose({ outcome: "error", status_code: 500 }).kind, "error");
  assert.equal(diagnose({ outcome: "cancelled" }).kind, "cancelled");
});

test("balanced when no stage reaches half of the total", () => {
  const c = diagnose({
    offsets_ms: { admit: 3000, first_token: 6000, done: 9500 },
  });
  assert.equal(c.kind, "balanced");
});

test("slow rule: fixed thresholds below 20 samples, p90 from 20", () => {
  const few = Array.from({ length: 5 }, () => ({ ttft_ms: 100 }));
  assert.deepEqual(slowRule(few), {
    ttftMs: SLOW_TTFT_MS,
    totalMs: SLOW_TOTAL_MS,
    basis: "fixed",
  });
  const many = Array.from({ length: 20 }, (_, i) => ({
    ttft_ms: (i + 1) * 100,
  }));
  const rule = slowRule(many);
  assert.equal(rule.basis, "p90");
  assert.equal(
    rule.ttftMs,
    percentile(
      many.map((m) => m.ttft_ms),
      0.9,
    ),
  );
  assert.equal(many.filter((m) => isSlow(m, rule)).length, 2);
  assert.equal(isSlow({}, rule), false);
});

test("sorting puts rows without the number last in both directions", () => {
  const rows = [
    { id: "a", ttft_ms: 300 },
    { id: "b" },
    { id: "c", ttft_ms: 900 },
  ];
  assert.deepEqual(
    sortRows(rows, { key: "ttft", dir: "desc" }).map((r) => r.id),
    ["c", "a", "b"],
  );
  assert.deepEqual(
    sortRows(rows, { key: "ttft", dir: "asc" }).map((r) => r.id),
    ["a", "c", "b"],
  );
});
