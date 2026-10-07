import { expect, test, type Page } from "@playwright/test";

/**
 * Polling must change digits, never geometry. The status feed below swaps
 * values between frames (more or fewer digits, a missing speculative mode, a
 * request that appears and one that ends) and every box in the stable zones
 * must keep its size and position.
 */
const decode = [9.9, 10.1, 99.9, 100.2, 1234.5, 7, 0.5, 55.7];
function frame(i: number) {
  const n = i % decode.length;
  const items = Array.from({ length: [0, 3, 1, 5, 0, 2, 4, 1][n] }, (_, k) => ({
    request_id: `req_${"a".repeat(4 + ((n + k) % 20))}`,
    elapsed_s: [0.4, 12.3, 61, 3725, 9.9][(n + k) % 5],
    phase: ["decode", "prefill", "queued"][(n + k) % 3],
    prompt_tokens: 20 + n * 1000,
    cached_tokens: n * 111,
    completion_tokens: n * 7,
    tokens_per_second: k % 2 ? null : decode[n],
    model: "Qwen3.8-27B",
  }));
  return {
    object: "yunshu.status",
    version: "fixture-1.0",
    state: "running",
    uptime_s: [59, 61, 3599, 3601, 86400][n % 5] + i,
    load_error: null,
    models: [
      {
        id: "Qwen3.8-27B",
        type: "VLMEngine",
        loaded: true,
        loading: false,
        pinned: n % 2 === 0,
        size_gb: n % 2 ? 15.7 : 105.25,
        expires_in_s: n % 2 ? 599 : null,
      },
    ],
    memory: {
      active_gb: [9.9, 10.1, 99.9, 100.2, 24.8, 7, 0.5, 128][n],
      cache_gb: n,
      peak_gb: 30 + n,
      total_gb: 137.4,
    },
    requests: {
      active: items.length,
      queued: 0,
      prefill: 0,
      decode: items.length,
      items,
    },
    last: {
      request_id: "req_last",
      prompt_tokens: [5, 120, 4096, 131072, 12, 987654, 3, 70][n],
      completion_tokens: [1, 22, 300, 4000, 9, 77, 5, 8][n],
      cached_tokens: [0, 100, 4000, 131000, 10, 987000, 2, 70][n],
      prefill_tps: n % 3 ? [8, 120, 1234, 99999, 5, 45, 17, 800][n] : null,
      decode_tps: n % 4 ? decode[n] : null,
      ttft_ms: [9, 120, 1234, 12345, 87, 999, 5, 15000][n],
      t: 1_700_000_000 + i,
      speculative:
        n % 2 ? { mode: "mtp", acceptance_rate: n / 10, rounds: 4 } : null,
    },
    throughput: {
      window_s: 60,
      requests: [0, 9, 99, 100, 1234, 5, 77, 3][n],
      prompt_tokens: 1,
      completion_tokens: 1,
      live_decode_tps: n % 3 ? decode[n] : null,
      mean_prefill_tps: [8, 120, 1234, 99999, 5, 45, 17, 800][n],
      mean_decode_tps: decode[(n + 3) % 8],
    },
  };
}

async function install(page: Page) {
  let i = 0;
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    const ok = path === "/v1/yunshu/status";
    return route.fulfill({
      status: ok ? 200 : 404,
      contentType: "application/json",
      body: JSON.stringify(ok ? frame(i++) : { detail: "fixture" }),
    });
  });
}

const ZONES: Record<string, string> = {
  "top bar": "header",
  // Engine and memory always lead the band at a fixed width. 現在 (decode, prefill,
  // idle) changes width with the phase and the pills after it (GPU, 請求, 交換)
  // appear only when they have something to say, so they sit behind the stable pair.
  "status pills": "ul[aria-label='引擎狀態'] > li:nth-child(-n+2)",
  sidebar: "[aria-label='控制台導覽']",
  "stat tiles":
    "[data-testid='overview-stats'], [data-testid='request-stats'], [data-testid='resource-readouts'], [data-testid='models']",
};

/** Every non-inline box inside the zone, in DOM order: [tag, x, y, w, h]. */
function snapshot(page: Page, selector: string) {
  return page.evaluate((sel) => {
    const out: [string, number, number, number, number][] = [];
    for (const root of document.querySelectorAll(sel))
      for (const el of [root, ...root.querySelectorAll("*")]) {
        // Chart and bar content is data: it may move. Its frame may not.
        if (el.closest("svg")) continue;
        // A model icon loads lazily into a slot of fixed size: the slot is a box, its content is not.
        if (
          el.closest("[data-icon-slot]") &&
          !el.hasAttribute("data-icon-slot")
        )
          continue;
        const host = el.parentElement?.closest(
          "[role=img], [role=progressbar], [role=meter]",
        );
        if (host && root.contains(host)) continue;
        const style = getComputedStyle(el);
        if (style.display === "inline" || style.display === "none") continue;
        const r = el.getBoundingClientRect();
        out.push([
          el.tagName.toLowerCase() + "." + String(el.className).slice(0, 30),
          Math.round(r.x * 2) / 2,
          Math.round((r.y + scrollY) * 2) / 2,
          Math.round(r.width * 2) / 2,
          Math.round(r.height * 2) / 2,
        ]);
      }
    return out;
  }, selector);
}

for (const route of ["overview", "requests", "models", "diagnostics"]) {
  test(`polling changes digits, not geometry: ${route}`, async ({ page }) => {
    test.setTimeout(60_000);
    await install(page);
    await page.clock.install();
    await page.goto(`/console/#/${route}`, { waitUntil: "domcontentloaded" });
    await page.getByLabel("引擎狀態").waitFor();
    const readings: Record<string, string[]> = {};
    const frames: Record<string, Awaited<ReturnType<typeof snapshot>>[]> = {};
    for (const name of Object.keys(ZONES)) frames[name] = [];
    for (let k = 0; k < 8; k++) {
      await page.clock.runFor(3100);
      await page.waitForTimeout(150);
      for (const [name, selector] of Object.entries(ZONES))
        frames[name].push(await snapshot(page, selector));
      (readings.pills ??= []).push(
        (await page.getByLabel("引擎狀態").innerText()).replace(/\s+/g, " "),
      );
    }
    // The feed really changed what is on screen between frames.
    expect(new Set(readings.pills).size).toBeGreaterThan(3);
    for (const [name, list] of Object.entries(frames)) {
      expect(list[0].length, `${name} has boxes`).toBeGreaterThan(0);
      for (const f of list.slice(1)) {
        const names = (frame: typeof f) => frame.map((box) => box[0]);
        const extra = names(f).filter(
          (n, i, a) =>
            a.indexOf(n) === i &&
            names(f).filter((x) => x === n).length !==
              names(list[0]).filter((x) => x === n).length,
        );
        expect(
          f.length,
          `${name} keeps its box count (changed: ${extra.join(" | ")})`,
        ).toBe(list[0].length);
        // Half a pixel of rounding is allowed; a digit changing width is not.
        const moved = f.filter((box, k) =>
          box.some(
            (v, c) =>
              c > 0 &&
              Math.abs((v as number) - (list[0][k][c] as number)) > 0.6,
          ),
        );
        expect(
          moved.map((box) => box[0]),
          `${name} keeps every box where it was`,
        ).toEqual([]);
      }
    }
  });
}
