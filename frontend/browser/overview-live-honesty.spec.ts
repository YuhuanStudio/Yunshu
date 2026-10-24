import { expect, test, type Route } from "@playwright/test";

// Under agent load the Overview showed the client's model alias ("m"), a truncated prefill sentence, and
// divider lines for empty placeholder slots under the one real request.

function status(sample: number, alias: string) {
  return {
    object: "yunshu.status",
    version: "fixture-1.0",
    state: "ready",
    uptime_s: 3_600 + sample,
    load_error: null,
    models: [
      { id: "Qwen3.8-27B-oQ4e-mtp", type: "text", loaded: true, loading: false, pinned: true, size_gb: 8, expires_in_s: null },
    ],
    memory: { active_gb: 12, cache_gb: 2, peak_gb: 14, total_gb: 64, pressure: 0.2 },
    requests: {
      active: 1,
      queued: 0,
      prefill: 0,
      decode: 1,
      items: [
        {
          request_id: "req_live_decode_0001",
          phase: "decode",
          model: "Qwen3.8-27B-oQ4e-mtp",
          model_alias: alias,
          elapsed_s: 10 + sample,
          completion_tokens: 100 * sample,
          tokens_per_second: 40,
          offsets_ms: { arrive: 0, admit: 10, first_token: 200 },
        },
      ],
    },
    last: {
      request_id: "req_done_1",
      prompt_tokens: 1000,
      completion_tokens: 50,
      cached_tokens: 1000,
      prefill_tps: 5000,
      decode_tps: 40,
      ttft_ms: 200,
      t: Date.now() / 1000 - 5,
    },
    throughput: {
      window_s: 60,
      requests: 1,
      prompt_tokens: 1000,
      completion_tokens: 50,
      live_decode_tps: 40,
      mean_prefill_tps: 800,
      mean_decode_tps: 35,
    },
  };
}

test("one live request: served model name, short prefill state, no fake rows", async ({
  page,
}) => {
  let n = 0;
  const json = (route: Route, body: unknown) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  await page.route("**/v1/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") return json(route, status(++n, "m"));
    return route.fulfill({ status: 404, contentType: "application/json", body: "{}" });
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  const list = page.locator("[data-live-list]");
  await expect(list).toContainText("req_live_decode");
  await expect(list).toContainText("Qwen3.8-27B-oQ4e-mtp");
  await expect(list).not.toContainText(/·\s*m\b/);
  // Prefill card: a short state, not a sentence the card cuts off.
  const label = page.getByTestId("speed-prefill-label");
  await expect(label).toBeVisible();
  const clipped = await label.evaluate((el) => el.scrollWidth > el.clientWidth + 1 || (el.parentElement ? el.parentElement.scrollWidth > el.parentElement.clientWidth + 1 : false));
  expect(clipped).toBe(false);
  // Empty slots keep their height but draw no divider lines.
  const lines = await list.locator("> li").evaluateAll((els) =>
    els.map((el) => ({
      hidden: el.getAttribute("aria-hidden") === "true",
      border: parseFloat(getComputedStyle(el).borderBottomWidth),
    })),
  );
  expect(lines.some((l) => l.hidden)).toBe(true);
  for (const l of lines) expect(l.border).toBe(0);
});
