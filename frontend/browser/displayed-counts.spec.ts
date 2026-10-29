import { expect, test, type Route } from "@playwright/test";

// A number the console presents as a fact must equal the engine's value at its sample time: counts
// snap to the sampled value, they never glide through values the engine did not report.

const MODEL = "Qwen3.5-2B-4bit";

function status(n: number) {
  return {
    object: "yunshu.status",
    version: "fixture-1.0",
    state: "ready",
    uptime_s: 3_600 + n,
    load_error: null,
    models: [
      {
        id: MODEL,
        type: "text",
        loaded: true,
        loading: false,
        pinned: true,
        size_gb: 2,
        expires_in_s: null,
      },
    ],
    memory: {
      active_gb: 2 + n * 0.37,
      cache_gb: 1,
      peak_gb: 6,
      total_gb: 64,
      pressure: 0.2,
    },
    requests: {
      active: 1,
      queued: 0,
      prefill: 0,
      decode: 1,
      items: [
        {
          request_id: "req_honest_decode_0001",
          phase: "decode",
          model: MODEL,
          elapsed_s: 5 + n,
          prompt_tokens: 800,
          cached_tokens: 0,
          completion_tokens: 36 + 240 * n,
          tokens_per_second: 40,
          offsets_ms: { arrive: 0, admit: 10, first_token: 200 },
        },
      ],
    },
    last: null,
    throughput: {
      window_s: 60,
      requests: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      live_decode_tps: 40,
      mean_prefill_tps: null,
      mean_decode_tps: 35,
    },
  };
}

test("the live token count is always a count the engine reported, within one readout interval", async ({
  page,
}) => {
  let n = 0;
  const json = (route: Route, body: unknown) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  await page.route("**/v1/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") return json(route, status(++n));
    return route.fulfill({
      status: 404,
      contentType: "application/json",
      body: "{}",
    });
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  const list = page.locator("[data-live-list]");
  await expect(list).toContainText("req_honest_decode");
  const shown = async () => {
    const text = (await list.innerText()).replace(/,/g, "");
    const m = /(\d+)\s*tok(?!\/)/.exec(text);
    return m ? Number(m[1]) : null;
  };
  let checked = 0;
  let seen = 0;
  for (let i = 0; i < 40 && checked < 6; i++) {
    const now = n;
    if (now === seen || now === 0) {
      await page.waitForTimeout(40);
      continue;
    }
    seen = now;
    // A readout is drawn at most every ~240 ms, so it may trail the newest sample by one interval; what
    // it shows is always a count the engine reported (never an in-between value).
    await page.waitForTimeout(280);
    const reported = (k: number) => 36 + 240 * k;
    const value = await shown();
    const k = Math.round(((value ?? NaN) - 36) / 240);
    expect(value, `sample ${now}`).toBe(reported(k));
    expect(k, `sample ${now} shows an old count`).toBeGreaterThanOrEqual(
      now - 3,
    );
    expect(k).toBeLessThanOrEqual(n);
    checked++;
  }
  expect(checked).toBeGreaterThanOrEqual(4);
});
