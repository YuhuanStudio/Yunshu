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
