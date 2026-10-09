import assert from "node:assert/strict";
import test from "node:test";
import {
  BACK_HOLD_MS,
  IDLE_HOLD_MS,
  emaStep,
  initialStablePhase,
  nextStablePhase,
} from "../src/phase-stability.ts";
import type { ActivityPhase } from "../src/engineView.ts";

const run = (seq: [ActivityPhase, number][]) => {
  let s = initialStablePhase(seq[0][0]);
  const out: ActivityPhase[] = [];
  for (const [raw, t] of seq) {
    s = nextStablePhase(s, raw, t);
    out.push(s.phase);
  }
  return out;
};

test("forward moves are immediate: idle, queued, prefill, decode", () => {
  assert.deepEqual(
    run([
      ["idle", 0],
      ["queued", 100],
      ["prefill", 200],
      ["decode", 300],
    ]),
    ["idle", "queued", "prefill", "decode"],
  );
});

test("a short prefill between two decodes does not flash 預填", () => {
  // One agent: decode, a 300 ms prefill of the next turn, decode again.
  const out = run([
    ["decode", 0],
    ["prefill", 250],
    ["prefill", 500],
    ["decode", 550],
    ["decode", 800],
  ]);
  assert.deepEqual(out, ["decode", "decode", "decode", "decode", "decode"]);
});

test("a prefill that lasts is shown once it has held", () => {
  const out = run([
    ["decode", 0],
    ["prefill", 100],
    ["prefill", 100 + BACK_HOLD_MS - 1],
    ["prefill", 100 + BACK_HOLD_MS],
  ]);
  assert.deepEqual(out, ["decode", "decode", "decode", "prefill"]);
});

test("the gap between two requests does not flash idle, a real stop does", () => {
  const out = run([
    ["decode", 0],
    ["idle", 200],
    ["decode", 700],
    ["idle", 1000],
    ["idle", 1000 + IDLE_HOLD_MS],
  ]);
  assert.deepEqual(out, ["decode", "decode", "decode", "decode", "idle"]);
});

test("a single request never alternates: prefill then decode only moves forward", () => {
  const raws: ActivityPhase[] = [
    "queued",
    "prefill",
    "prefill",
    "decode",
    "decode",
    "decode",
    "idle",
  ];
  let s = initialStablePhase("idle");
  const seen: ActivityPhase[] = ["idle"];
  raws.forEach((raw, i) => {
    s = nextStablePhase(s, raw, i * 250);
    if (seen.at(-1) !== s.phase) seen.push(s.phase);
  });
  assert.deepEqual(seen, ["idle", "queued", "prefill", "decode"]);
});

test("ema: first sample as is, null clears, later samples move toward the new value", () => {
  assert.equal(emaStep(null, 50, 250), 50);
  assert.equal(emaStep(50, null, 250), null);
  const v = emaStep(50, 70, 250, 1000)!;
  assert.ok(v > 50 && v < 70);
  assert.ok(Math.abs(emaStep(50, 70, 10_000, 1000)! - 70) < 0.01);
});
