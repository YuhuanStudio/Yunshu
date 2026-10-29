import type { Page } from "@playwright/test";
import { FIXTURES } from "./real-load";

// A storm: far busier than the recorded real stream. A request starts every ~400 ms and lives about
// 2.4 s (prefill for the first 0.8 s, then decode), so six are in flight and one arrives and one
// ends every 400 ms; speeds and memory jitter on every poll; a finished request lands in the
// recent list every 300 ms. Built from a real status/recent row so every field has its real shape.
type Obj = Record<string, unknown>;
const lines = FIXTURES.burst;
const baseStatus = lines.filter((l) => l.status)[40].status as Obj;
const baseRow = lines.find((l) => l.requests)!.requests![0] as Obj;
const baseItem = (baseStatus.requests as { items: Obj[] }).items[0];

const LIFE = 2400;
const EVERY = 400;
const PREFILL = 800;
const noise = (ms: number, k: number) =>
  Math.abs(Math.sin(ms / 97 + k * 12.9898) * 43758.5453) % 1;

export function stormStatus(ms: number): Obj {
  const first = Math.floor(ms / EVERY);
  const items: Obj[] = [];
  for (let k = 0; k < LIFE / EVERY; k++) {
    const n = first - k;
    if (n < 0) continue;
    const age = ms - n * EVERY;
    if (age >= LIFE) continue;
    const prefill = age < PREFILL;
    items.push({
      ...baseItem,
      request_id: `req_storm_${n}`,
      elapsed_s: age / 1000,
      phase: prefill ? "prefill" : "decode",
      prompt_tokens: 3000 + (n % 5) * 1000,
      cached_tokens: (n % 3) * 500,
      processed_tokens: prefill ? Math.round((age / PREFILL) * 3000) : 3000,
      percent: prefill ? Math.round((age / PREFILL) * 100) : null,
      completion_tokens: prefill
        ? 0
        : Math.round(((age - PREFILL) / 1000) * 80),
      tokens_per_second: prefill
        ? 900 + noise(ms, n) * 200
        : 60 + noise(ms, n) * 80,
    });
  }
  const decode = items.filter((i) => i.phase === "decode").length;
  const mem = baseStatus.memory as Obj;
  return {
    ...baseStatus,
    memory: { ...mem, active_gb: 9 + noise(ms, 99) * 4 },
    requests: {
      active: items.length,
      // a queue that drifts across the 9 / 10 boundary, not a new random depth on every poll
      queued: Math.round(9 + 2.5 * Math.sin(ms / 3100) + noise(ms, 7) * 1.4),
      prefill: items.length - decode,
      decode,
      items,
    },
    throughput: {
      ...(baseStatus.throughput as Obj),
      live_decode_tps: decode ? 60 + noise(ms, 5) * 80 : null,
    },
  };
}

export function stormRecent(ms: number): Obj[] {
  const n = Math.floor(ms / 300);
  const rows: Obj[] = [];
  for (let i = 0; i < 40 && n - i >= 0; i++)
    rows.push({
      ...baseRow,
      request_id: `req_done_${n - i}`,
      t: Date.now() / 1000 - (ms - (n - i) * 300) / 1000,
    });
  return rows;
}

export async function storm(page: Page, recorder: string) {
  const t0 = Date.now();
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.addInitScript(recorder);
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    const ms = Date.now() - t0;
    if (path === "/v1/yunshu/status")
      return route.fulfill({ json: stormStatus(ms) });
    if (path === "/v1/yunshu/requests/recent") {
      const rows = stormRecent(ms);
      return route.fulfill({
        json: {
          object: "list",
          data: rows,
          count: rows.length,
          capacity: 512,
          boot_id: "storm",
          latest_seq: Math.floor(ms / 300),
        },
      });
    }
    return route.fulfill({ status: 404, json: {} });
  });
}
