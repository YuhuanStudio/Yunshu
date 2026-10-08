import assert from "node:assert/strict";
import test from "node:test";
import { binaryGb, bytesText, splitBytes } from "../src/byte-format.ts";
import { parseMemory } from "../src/memory-api.ts";
import { parseEngineStatus } from "../src/api.ts";

test("a 128 GiB machine reads 128 GB, as macOS shows it", () => {
  assert.equal(bytesText(137438953472), "128GB");
  assert.deepEqual(splitBytes(137438953472), { value: "128", unit: "GB" });
  assert.equal(splitBytes(1024 * 1024 * 5).unit, "MB");
});

test("the engine's decimal *_gb fields are converted once, at the parse boundary", () => {
  assert.ok(Math.abs(binaryGb(137.438953472) - 128) < 1e-9);
  const m = parseMemory({
    total_gb: 137.438953472,
    mlx: { active_gb: 21.47483648 },
  });
  assert.ok(Math.abs((m.total_gb as number) - 128) < 1e-9);
  assert.ok(Math.abs((m.mlx.active_gb as number) - 20) < 1e-9);
});

test("status memory and model sizes are binary too", () => {
  const s = parseEngineStatus({
    object: "yunshu.status",
    version: "x",
    state: "ok",
    uptime_s: 1,
    load_error: null,
    models: [
      {
        id: "m",
        type: "LLM",
        loaded: true,
        loading: false,
        pinned: false,
        size_gb: 10.73741824,
      },
    ],
    memory: { total_gb: 137.438953472, active_gb: 1, pressure: 0.2 },
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
  });
  assert.ok(Math.abs((s.memory.total_gb as number) - 128) < 1e-9);
  assert.ok(Math.abs((s.models[0].size_gb as number) - 10) < 1e-9);
  assert.equal(s.memory.pressure, 0.2);
});

test("configured limits are shown exactly as entered (the engine reads them as GiB), never converted", () => {
  const m = parseMemory({ limits: { apc_max_gb: 8, apc_warm_max_gb: 4 } });
  assert.equal(m.limits.apc_max_gb, 8);
  assert.equal(m.limits.apc_warm_max_gb, 4);
});

import { gbTotalText, memoryPairText } from "../src/byte-format.ts";
test("one used / total rule: spaces around the slash, whole totals without decimals", () => {
  assert.equal(memoryPairText(23.2, 128), "23.2 / 128 GB");
  assert.equal(memoryPairText(23.24, 127.96), "23.2 / 128 GB");
  assert.equal(gbTotalText(63.5), "63.5");
});

import { parseServerHistory } from "../src/history-api.ts";
import { readGb } from "../src/byte-format.ts";

const GiB = 2 ** 30;
const statusWith = (
  memory: Record<string, unknown>,
  model: Record<string, unknown>,
) =>
  parseEngineStatus({
    object: "yunshu.status",
    version: "x",
    state: "ok",
    uptime_s: 1,
    load_error: null,
    models: [
      {
        id: "m",
        type: "LLM",
        loaded: true,
        loading: false,
        pinned: false,
        ...model,
      },
    ],
    memory,
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
  });

test("newer engine: *_bytes win and binary *_gb are never converted twice", () => {
  const s = statusWith(
    {
      total_gb: 128,
      total_bytes: 128 * GiB,
      active_gb: 20,
      active_bytes: 20 * GiB,
    },
    { size_gb: 10, size_bytes: 10 * GiB },
  );
  assert.equal(s.memory.total_gb, 128);
  assert.equal(s.memory.active_gb, 20);
  assert.equal(s.models[0].size_gb, 10);
  assert.equal(readGb({ x_gb: 128 }, "x"), 128 * (1e9 / GiB)); // legacy shape converts once
});

test("legacy engine (0.1.4): decimal *_gb without bytes converts once", () => {
  const s = statusWith(
    { total_gb: 137.438953472, active_gb: 21.47483648 },
    { size_gb: 10.73741824 },
  );
  assert.ok(Math.abs((s.memory.total_gb as number) - 128) < 1e-9);
  assert.ok(Math.abs((s.memory.active_gb as number) - 20) < 1e-9);
  assert.ok(Math.abs((s.models[0].size_gb as number) - 10) < 1e-9);
});

test("ledger and history work for both generations", () => {
  const fresh = parseMemory({
    total_gb: 128,
    total_bytes: 128 * GiB,
    free_gb: 60,
    free_bytes: 60 * GiB,
    mlx: { active_gb: 20, active_bytes: 20 * GiB },
    owners: [{ kind: "weights", bytes: 10 * GiB, gb: 10 }],
  });
  assert.equal(fresh.total_gb, 128);
  assert.equal(fresh.free_gb, 60);
  assert.equal(fresh.mlx.active_gb, 20);
  assert.equal(fresh.owners[0].gb, 10);
  const old = parseMemory({
    total_gb: 137.438953472,
    free_gb: 64.42450944,
    owners: [{ kind: "weights", bytes: 10 * GiB, gb: 10.73741824 }],
  });
  assert.ok(Math.abs((old.total_gb as number) - 128) < 1e-9);
  assert.ok(Math.abs((old.free_gb as number) - 60) < 1e-9);
  const rows = {
    t: [1, 2, 3, 4, 5, 6],
    active_gb: [20, 20, 20, 20, 20, 20],
    cache_gb: [2, 2, 2, 2, 2, 2],
  };
  const binaryPts = parseServerHistory({ series: rows, interval_s: 1 }, true);
  const legacyPts = parseServerHistory({ series: rows, interval_s: 1 }, false);
  assert.ok(binaryPts && legacyPts);
  assert.equal(binaryPts.points[0].memActive, 20);
  assert.ok(
    Math.abs((legacyPts.points[0].memActive as number) - 20 * (1e9 / GiB)) <
      1e-9,
  );
});
