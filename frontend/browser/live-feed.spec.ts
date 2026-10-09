import { expect, test, type Page } from "@playwright/test";

type Items = Record<string, unknown>[];
const status = (items: Items, extra: Record<string, unknown> = {}) => ({
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "ready",
  uptime_s: 100,
  load_error: null,
  models: [
    {
      id: "Qwen-VL",
      type: "VLMEngine",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 16,
    },
  ],
  memory: { active_gb: 12, cache_gb: 2, peak_gb: 14, total_gb: 64 },
  requests: {
    active: items.length,
    queued: 0,
    prefill: items.filter((i) => i.phase === "prefill").length,
    decode: items.filter((i) => i.phase === "decode").length,
    items,
  },
  last: null,
  throughput: {
    window_s: 60,
    requests: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    live_decode_tps: items.some((i) => i.phase === "decode") ? 60 : null,
    mean_prefill_tps: null,
    mean_decode_tps: null,
  },
  ...extra,
});

async function install(page: Page, feed: (t: number) => Items) {
  const t0 = Date.now();
  let polls = 0;
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") {
      polls += 1;
      return route.fulfill({ json: status(feed(Date.now() - t0)) });
    }
    return route.fulfill({ status: 404, json: { detail: "Not Found" } });
  });
  return { polls: () => polls };
}

test("an agent's short prefills between decodes never flip the phase back to 預填", async ({
  page,
}) => {
  // 900 ms decode, then a 300 ms prefill of the next turn, over and over: one request at a time.
  await install(page, (t) => {
    const cycle = t % 1200;
    return cycle < 900
      ? [
          {
            request_id: "r",
            elapsed_s: 1,
            phase: "decode",
            completion_tokens: 50 + Math.floor(t / 20),
            tokens_per_second: 60,
          },
        ]
      : [
          {
            request_id: "r2",
            elapsed_s: 0.3,
            phase: "prefill",
            prompt_tokens: 3000,
            cached_tokens: 2900,
            processed_tokens: 2950,
            percent: 50,
          },
        ];
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  const pill = page.getByTestId("live-phase");
  await expect(pill).toHaveAttribute("data-phase", /解碼/);
  const seen = new Set<string>();
  for (let i = 0; i < 40; i++) {
    seen.add((await pill.getAttribute("data-phase")) ?? "");
    await page.waitForTimeout(120);
  }
  expect([...seen]).toEqual(["解碼"]);
});

test("a single request goes prefill then decode and never back", async ({
  page,
}) => {
  await install(page, (t) =>
    t < 1500
      ? [
          {
            request_id: "r",
            elapsed_s: t / 1000,
            phase: "prefill",
            prompt_tokens: 9000,
            cached_tokens: 0,
            processed_tokens: Math.round(t * 5),
            percent: Math.min(99, t / 15),
          },
        ]
      : [
          {
            request_id: "r",
            elapsed_s: t / 1000,
            phase: "decode",
            completion_tokens: Math.round((t - 1500) / 20),
            tokens_per_second: 55,
          },
        ],
  );
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  const pill = page.getByTestId("live-phase");
  const order: string[] = [];
  for (let i = 0; i < 40; i++) {
    const p = (await pill.getAttribute("data-phase")) ?? "";
    if (order.at(-1) !== p) order.push(p);
    await page.waitForTimeout(100);
  }
  const filtered = order.filter((p) => /預填|解碼/.test(p));
  expect(filtered.length).toBeLessThanOrEqual(2);
  expect(filtered.at(-1)).toMatch(/解碼/);
  if (filtered.length === 2) expect(filtered[0]).toMatch(/預填/);
});

test("a cache hit is its own segment in the prefill bar, then the computed part fills in after it", async ({
  page,
}) => {
  await install(page, () => [
    {
      request_id: "r1",
      elapsed_s: 2,
      phase: "prefill",
      model: "Qwen-VL",
      prompt_tokens: 10000,
      cached_tokens: 8000,
      processed_tokens: 9000,
      percent: 50,
      tokens_per_second: 800,
    },
  ]);
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  const bar = page.getByRole("progressbar", { name: "預填進度" }).first();
  await expect(bar).toBeVisible();
  await expect(bar).toHaveAttribute("title", /快取命中 8,000 tokens/);
  await expect(bar).toHaveAttribute(
    "aria-valuetext",
    /快取 8,000 \+ 已計算 1,000 \/ 共 10,000 tokens/,
  );
  const segs = bar.locator("span");
  await expect(segs.nth(0)).toHaveAttribute("data-tone", "info");
  // 80% of the prompt is the cache hit: the bar does not start part-way along by being drawn from there.
  await expect
    .poll(
      async () =>
        (await segs.nth(0).evaluate((e) => e.getBoundingClientRect().width)) /
        (await bar.evaluate((e) => e.getBoundingClientRect().width)),
    )
    .toBeCloseTo(0.8, 1);
  await expect(segs.nth(1)).toHaveAttribute("data-tone", "neutral");
  await expect
    .poll(
      async () =>
        (await segs.nth(1).evaluate((e) => e.getBoundingClientRect().width)) /
        (await bar.evaluate((e) => e.getBoundingClientRect().width)),
    )
    .toBeCloseTo(0.1, 1);
});

test("the status is read about four times a second while a request runs and calmly when idle", async ({
  page,
}) => {
  let busy = false;
  const probe = await install(page, () =>
    busy
      ? [
          {
            request_id: "r",
            elapsed_s: 1,
            phase: "decode",
            completion_tokens: 10,
            tokens_per_second: 50,
          },
        ]
      : [],
  );
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(500);
  const idle0 = probe.polls();
  await page.waitForTimeout(2600);
  expect(probe.polls() - idle0).toBeLessThanOrEqual(2);
  busy = true;
  await page.waitForTimeout(3600); // at most one calm interval to notice the work
  const busy0 = probe.polls();
  await page.waitForTimeout(2000);
  expect(probe.polls() - busy0).toBeGreaterThanOrEqual(5);
});
