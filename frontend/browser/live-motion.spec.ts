import { expect, test, type Page } from "@playwright/test";

// Live data is real-time and smooth: numbers glide to each sample, progress bars move between
// samples, the sparkline slides with the wall clock. Under reduced motion every one of them draws
// the samples as they are, so these tests can tell the two apart by watching frames.

const memory = { active_gb: 12, cache_gb: 2, peak_gb: 14, total_gb: 64 };
const status = (t: number) => {
  const tps = Math.floor(t / 600) % 2 === 0 ? 40 : 80;
  const prefilling = t < 6000;
  const items = prefilling
    ? [
        {
          request_id: "r1",
          elapsed_s: t / 1000,
          phase: "prefill",
          prompt_tokens: 9000,
          cached_tokens: 0,
          processed_tokens: Math.round(t * 1.5),
          percent: Math.min(99, t / 60),
        },
      ]
    : [
        {
          request_id: "r1",
          elapsed_s: t / 1000,
          phase: "decode",
          completion_tokens: Math.round((t - 6000) / 20),
          tokens_per_second: tps,
        },
      ];
  return {
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
    memory,
    requests: {
      active: 1,
      queued: 0,
      prefill: prefilling ? 1 : 0,
      decode: prefilling ? 0 : 1,
      items,
    },
    last: null,
    throughput: {
      window_s: 60,
      requests: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      live_decode_tps: prefilling ? null : tps,
      mean_prefill_tps: null,
      mean_decode_tps: null,
    },
  };
};

async function open(page: Page, hash: string, startAt = 0) {
  const t0 = Date.now() - startAt;
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status")
      return route.fulfill({ json: status(Date.now() - t0) });
    return route.fulfill({ status: 404, json: { detail: "Not Found" } });
  });
  await page.goto(`/console/#/${hash}`, { waitUntil: "domcontentloaded" });
}

/** The text of `selector` on every animation frame for `ms`. */
const watch = (page: Page, selector: string, ms: number, read: string) =>
  page.evaluate(
    ([sel, duration, code]) =>
      new Promise<string[]>((resolve) => {
        const out: string[] = [];
        const end = performance.now() + (duration as number);
        const tick = () => {
          const el = document.querySelector(sel as string);
          if (el) out.push(new Function("el", `return ${code}`)(el));
          if (performance.now() < end) requestAnimationFrame(tick);
          else resolve(out);
        };
        requestAnimationFrame(tick);
      }),
    [selector, ms, read] as const,
  );

const DECODE = '[data-testid="speed-decode"] .tabular-nums';
const BAR_FILL = '[data-yunui="live-bar"] > span:last-child';
const scale =
  "Number(/scaleX\\(([^)]+)\\)/.exec(el.style.transform)?.[1] ?? 0)";

test("numbers glide between samples (and only then)", async ({ page }) => {
  await open(page, "overview", 6500);
  await expect(page.locator(DECODE).first()).toBeVisible();
  const seen = new Set(await watch(page, DECODE, 3000, "el.textContent"));
  const values = [...seen]
    .map((v) => Number(v.replace(/[^\d.]/g, "")))
    .filter((n) => Number.isFinite(n));
  const between = values.filter((v) => v > 42 && v < 78);
  expect(between.length, `values seen: ${[...seen].join(" ")}`).toBeGreaterThan(
    3,
  );
  expect(Math.min(...values)).toBeGreaterThanOrEqual(40);
  expect(Math.max(...values)).toBeLessThanOrEqual(80);
});

test("reduced motion draws each sample as it is: no in-between values", async ({
  browser,
}) => {
  const ctx = await browser.newContext({ reducedMotion: "reduce" });
  const page = await ctx.newPage();
  await open(page, "overview", 6500);
  await expect(page.locator(DECODE).first()).toBeVisible();
  const seen = new Set(await watch(page, DECODE, 3000, "el.textContent"));
  const values = [...seen]
    .map((v) => Number(v.replace(/[^\d.]/g, "")))
    .filter((n) => Number.isFinite(n));
  expect(values.filter((v) => v > 42 && v < 78)).toEqual([]);
  await ctx.close();
});

test("a running prefill bar moves on (almost) every frame, never backwards", async ({
  page,
}) => {
  await open(page, "overview", 500);
  await expect(page.locator(BAR_FILL).first()).toBeVisible();
  const frames = (await watch(page, BAR_FILL, 2200, scale)).map(Number);
  for (let i = 1; i < frames.length; i++)
    expect(frames[i]).toBeGreaterThanOrEqual(frames[i - 1] - 1e-9);
  const distinct = new Set(frames.map((f) => f.toFixed(5))).size;
  // 4 samples a second would give about 9 distinct values in 2.2 s; continuous motion gives dozens
  expect(
    distinct,
    `${distinct} distinct of ${frames.length}: ${frames
      .slice(0, 30)
      .map((f) => f.toFixed(3))
      .join(" ")}`,
  ).toBeGreaterThan(15);
  expect(frames.at(-1)!).toBeLessThanOrEqual(1);
});

test("reduced motion: the bar only changes when a sample arrives", async ({
  browser,
}) => {
  const ctx = await browser.newContext({ reducedMotion: "reduce" });
  const page = await ctx.newPage();
  await open(page, "overview", 500);
  await expect(page.locator(BAR_FILL).first()).toBeVisible();
  const frames = (await watch(page, BAR_FILL, 2200, scale)).map(Number);
  expect(new Set(frames.map((f) => f.toFixed(5))).size).toBeLessThan(16);
  await ctx.close();
});

test("a new request lowers the bar at once instead of running backwards", async ({
  page,
}) => {
  let n = 0;
  const t0 = Date.now();
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path !== "/v1/yunshu/status")
      return route.fulfill({ status: 404, json: {} });
    const t = Date.now() - t0;
    const id = t < 1800 ? "a" : "b";
    const base = id === "a" ? t : t - 1800;
    n++;
    const s = status(0);
    s.requests.items = [
      {
        request_id: id,
        elapsed_s: base / 1000,
        phase: "prefill",
        prompt_tokens: 4000,
        cached_tokens: 0,
        processed_tokens: Math.min(3900, Math.round(base * 2)),
        percent: Math.min(97, base / 20),
      } as never,
    ];
    return route.fulfill({ json: s });
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await expect(page.locator(BAR_FILL).first()).toBeVisible();
  const frames = (await watch(page, BAR_FILL, 3200, scale)).map(Number);
  const peak = Math.max(...frames.slice(0, Math.floor(frames.length / 2)));
  const drop = frames.findIndex((f, i) => i > 0 && f < frames[i - 1] - 0.2);
  expect(n).toBeGreaterThan(5);
  expect(peak).toBeGreaterThan(0.5);
  expect(
    drop,
    "the reset to the new request is one step down, not a slide",
  ).toBeGreaterThan(0);
  expect(frames[drop + 1] - frames[drop]).toBeLessThan(0.2);
});

test("the island sparkline slides with the clock between samples", async ({
  page,
}) => {
  await open(page, "overview", 7000);
  await expect(page.locator(DECODE).first()).toBeVisible();
  await page.waitForTimeout(13_500); // the series keeps a sample every 2.5 s; five are needed
  await page.getByTestId("footer-trigger").click();
  const spark = page.locator('[data-yunui="live-sparkline"]');
  await expect(spark).toBeVisible();
  const xs = (
    await watch(
      page,
      '[data-yunui="live-sparkline"] > g',
      900,
      "el.style.transform",
    )
  ).map((s) => Number(/translateX\(([-\d.]+)px\)/.exec(s)?.[1] ?? NaN));
  expect(new Set(xs).size).toBeGreaterThan(10);
});

test.describe("phone", () => {
  test.use({
    viewport: { width: 402, height: 874 },
    hasTouch: true,
    isMobile: true,
  });
  test("the animation loop keeps the frame budget", async ({ page }) => {
    await open(page, "overview", 500);
    await expect(page.locator(BAR_FILL).first()).toBeVisible();
    const deltas = await page.evaluate(
      () =>
        new Promise<number[]>((resolve) => {
          const out: number[] = [];
          let last = performance.now();
          const end = last + 2500;
          const tick = (t: number) => {
            out.push(t - last);
            last = t;
            if (t < end) requestAnimationFrame(tick);
            else resolve(out);
          };
          requestAnimationFrame(tick);
        }),
    );
    deltas.shift();
    const sorted = [...deltas].sort((a, b) => a - b);
    const p95 = sorted[Math.floor(sorted.length * 0.95)];
    expect(p95, `p95 frame ${p95.toFixed(1)} ms`).toBeLessThan(40);
    expect(deltas.filter((d) => d > 120)).toEqual([]);
  });
});

test("the hero chart's series slide continuously and re-base when a sample lands", async ({
  page,
}) => {
  test.setTimeout(60_000);
  await open(page, "overview", 7000);
  await expect(page.locator(DECODE).first()).toBeVisible();
  await page.waitForTimeout(14_000); // a sample every 2.5 s; the chart needs several
  const box = page.locator(".live-scroll").first();
  await expect(box).toBeVisible();
  const shifts = (
    await watch(
      page,
      ".live-scroll",
      6000,
      'Number(el.style.getPropertyValue("--live-shift") || 0)',
    )
  ).map(Number);
  expect(
    new Set(shifts.map((s) => s.toFixed(1))).size,
    "moves between samples, not in steps",
  ).toBeGreaterThan(30);
  expect(Math.max(...shifts)).toBeGreaterThan(1);
  expect(
    shifts.some((s, i) => i > 0 && s < shifts[i - 1] - 0.5),
    "re-bases when a sample lands",
  ).toBe(true);
  expect(Math.max(...shifts)).toBeLessThan(1100);
  const moved = await page.evaluate(() => {
    const el = document.querySelector(".live-scroll svg [clip-path] > *");
    return el ? getComputedStyle(el).transform : "none";
  });
  expect(moved).not.toBe("none");
});

test("reduced motion: the hero chart does not slide", async ({ browser }) => {
  test.setTimeout(60_000);
  const ctx = await browser.newContext({ reducedMotion: "reduce" });
  const page = await ctx.newPage();
  await open(page, "overview", 7000);
  await expect(page.locator(DECODE).first()).toBeVisible();
  await page.waitForTimeout(14_000);
  const shifts = (
    await watch(
      page,
      ".live-scroll",
      2000,
      'Number(el.style.getPropertyValue("--live-shift") || 0)',
    )
  ).map(Number);
  expect(Math.max(...shifts)).toBe(0);
  await ctx.close();
});

test("a request that ends fades out, then collapses without moving what is below", async ({
  page,
}) => {
  test.setTimeout(45_000);
  let ended = false;
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path !== "/v1/yunshu/status")
      return route.fulfill({ status: 404, json: {} });
    const s = status(8000);
    s.requests.items = (
      !ended
        ? [
            {
              request_id: "keep",
              elapsed_s: 2,
              phase: "decode",
              completion_tokens: 40,
              tokens_per_second: 50,
            },
            {
              request_id: "gone",
              elapsed_s: 2,
              phase: "decode",
              completion_tokens: 40,
              tokens_per_second: 50,
            },
          ]
        : [
            {
              request_id: "keep",
              elapsed_s: 2,
              phase: "decode",
              completion_tokens: 40,
              tokens_per_second: 50,
            },
          ]
    ) as never;
    s.requests.active = s.requests.items.length;
    return route.fulfill({ json: s });
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  const rows = page.locator("li.live-row");
  await expect(rows).toHaveCount(2);
  const pending = page.evaluate(
    () =>
      new Promise<{ n: number; leaving: number; op: number; h: number }[]>(
        (resolve) => {
          const out: { n: number; leaving: number; op: number; h: number }[] =
            [];
          const end = performance.now() + 2500;
          const tick = () => {
            const all = [...document.querySelectorAll("li.live-row")];
            const gone = all.find((r) => r.hasAttribute("data-leaving"));
            out.push({
              n: all.length,
              leaving: gone ? 1 : 0,
              op: gone
                ? Number(getComputedStyle(gone.firstElementChild!).opacity)
                : 1,
              h: gone ? gone.getBoundingClientRect().height : 0,
            });
            if (performance.now() < end) requestAnimationFrame(tick);
            else resolve(out);
          };
          requestAnimationFrame(tick);
        },
      ),
  );
  await page.waitForTimeout(300);
  ended = true;
  const samples = await pending;
  const leaving = samples.filter((s) => s.leaving);
  expect(
    leaving.length,
    "the ended row lingers while it leaves",
  ).toBeGreaterThan(5);
  expect(Math.min(...leaving.map((s) => s.op)), "it fades").toBeLessThan(0.5);
  const hs = leaving.map((s) => s.h);
  expect(hs[0], "full height while fading").toBeGreaterThan(20);
  expect(hs.at(-1)!, "collapsed after the fade").toBeLessThan(hs[0]);
  expect(samples.at(-1)!.n, "and then gone").toBe(1);
});
