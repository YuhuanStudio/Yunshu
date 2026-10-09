import assert from "node:assert/strict";
import test from "node:test";
import { parseEngineStatus, type EngineStatus } from "../src/api.ts";
import { offlineCause } from "../src/errors.ts";
import {
  footerPills,
  gpuBusyFraction,
  uptimeText,
} from "../src/footer-status.ts";
import { parseMemory } from "../src/memory-api.ts";

const mk = (over: Record<string, unknown> = {}) =>
  parseEngineStatus({
    object: "yunshu.status",
    version: "t",
    state: "running",
    uptime_s: 7300,
    load_error: null,
    models: [
      {
        id: "/m/Qwen3.8-27B",
        type: "VLM",
        loaded: true,
        loading: false,
        pinned: false,
      },
    ],
    memory: { active_gb: 42.94967296, total_gb: 137.438953472 },
    requests: { active: 0, queued: 0, prefill: 0, decode: 0, items: [] },
    last: {
      request_id: "r",
      prompt_tokens: 1,
      completion_tokens: 1,
      cached_tokens: 0,
      prefill_tps: 1,
      decode_tps: 9,
      ttft_ms: 330,
      t: 1,
    },
    throughput: {
      window_s: 60,
      requests: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      live_decode_tps: null,
      mean_decode_tps: 48,
      mean_prefill_tps: 700,
    },
    ...over,
  }) as EngineStatus;

const input = (status: EngineStatus | null, extra = {}) => ({
  phase: "online" as const,
  errorStatus: null,
  status,
  gpuBusy: null,
  ledger: null,
  ...extra,
});
const keys = (p: { key: string }[]) => p.map((x) => x.key);

test("idle online: engine, now and memory only; no last-request pills", () => {
  const pills = footerPills(input(mk()));
  assert.deepEqual(keys(pills), ["engine", "memory", "now"]);
  assert.equal(pills[0].value, "Qwen3.8-27B");
  assert.equal(pills[2].label, "閒置");
  assert.equal(pills[2].value, undefined);
  assert.ok(!JSON.stringify(pills).includes("TTFT"));
  assert.ok(pills[0].help.includes("2 小時 1 分"));
  // Running is the success tone on the dot; idle work has no dot and stays neutral.
  assert.equal(pills[0].tone, "success");
  assert.equal(pills[2].tone, "neutral");
  assert.equal(pills[2].dot, false);
});

test("decoding shows live tok/s and the active count", () => {
  const s = mk({
    requests: { active: 2, queued: 1, prefill: 0, decode: 2, items: [] },
    throughput: {
      window_s: 60,
      requests: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      live_decode_tps: 36.4,
      mean_decode_tps: 1,
      mean_prefill_tps: 1,
    },
  });
  const pills = footerPills(input(s));
  assert.equal(pills[2].label, "解碼");
  assert.equal(pills[2].value, "36 tok/s");
  assert.equal(pills.at(-1)?.key, "load");
  assert.equal(pills.at(-1)?.value, "2 進行 1 排隊");
});

test("memory tone follows pressure level, swap and gpu only when present", () => {
  const ledger = parseMemory({
    total_gb: 137.438953472,
    mlx: { active_gb: 42.94967296 },
    host: { pressure_level: "critical", swap_used_gb: 2.34 },
  });
  const pills = footerPills(input(mk(), { ledger, gpuBusy: 0.853 }));
  const byKey = Object.fromEntries(pills.map((p) => [p.key, p]));
  assert.equal(byKey.memory.tone, "danger");
  assert.match(byKey.memory.help, /嚴重/);
  assert.match(byKey.memory.value ?? "", /^40\.0 \/ 128 GB$/);
  assert.equal(byKey.swap.value, "2.2 GB");
  assert.equal(byKey.gpu.value, "85%");
  const calm = footerPills(
    input(mk(), {
      ledger: parseMemory({
        total_gb: 137.438953472,
        mlx: { active_gb: 42.94967296 },
        host: { pressure_level: "normal", swap_used_gb: 0 },
      }),
    }),
  );
  assert.deepEqual(keys(calm), ["engine", "memory", "now"]);
});

test("offline pills carry the cause; online-only pills disappear", () => {
  const down = footerPills({
    ...input(mk()),
    phase: "offline",
    errorStatus: 500,
  });
  assert.deepEqual(keys(down), ["engine"]);
  assert.equal(down[0].label, "離線");
  assert.equal(down[0].value, "內部錯誤");
  const proxyDown = footerPills({
    ...input(mk()),
    phase: "offline",
    errorStatus: 502,
  });
  assert.equal(proxyDown[0].value, "無法連線");
  assert.equal(down[0].tone, "danger");
  const unauthorized = footerPills({ ...input(null), phase: "unauthorized" });
  assert.equal(unauthorized[0].label, "未授權");
});

test("offlineCause titles", () => {
  assert.equal(offlineCause("offline", null).title, "無法連接引擎");
  assert.equal(offlineCause("offline", 503).title, "引擎內部錯誤");
  assert.equal(offlineCause("unauthorized", 401).title, "需要有效的存取權杖");
  assert.equal(offlineCause("offline", 404).title, "引擎回傳 HTTP 404");
  assert.equal(offlineCause("connecting", null).title, "正在連接引擎");
});

test("gpu busy fraction is a delta of the cumulative meters", () => {
  const g = (busy: number, up: number) => ({
    m: { slices: { busy_seconds: busy, uptime_seconds: up } },
  });
  const a = mk({ gpu: g(10, 100) });
  const b = mk({ gpu: g(16, 110) });
  assert.equal(gpuBusyFraction(a, b), 0.6);
  assert.equal(gpuBusyFraction(a, mk()), null);
  assert.equal(gpuBusyFraction(a, mk({ gpu: g(16, 100) })), null);
});

test("uptime text", () => {
  assert.equal(uptimeText(59), "59 秒");
  assert.equal(uptimeText(3725), "1 小時 2 分");
  assert.equal(uptimeText(90000), "1 天 1 小時");
});

test("memory GB comes from the status only; the ledger adds pressure, never a second figure", () => {
  const ledger = parseMemory({
    total_gb: 192,
    mlx: { active_gb: 99 },
    host: { pressure_level: "warn", swap_used_gb: 0 },
  });
  const memory = footerPills(input(mk(), { ledger })).find(
    (p) => p.key === "memory",
  );
  assert.equal(memory?.value, "40.0 / 128 GB");
  assert.equal(memory?.tone, "warning");
});
