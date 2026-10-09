import { devices, expect, test, type Page } from "@playwright/test";

test.use({
  ...devices["iPhone 15 Pro"],
  viewport: { width: 402, height: 874 },
});

const status = {
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
};

async function open(page: Page) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") return route.fulfill({ json: status });
    return route.fulfill({ status: 404, json: { detail: "Not Found" } });
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await page.getByTestId("footer-compact").click();
  await expect(page.getByRole("dialog", { name: "引擎狀態" })).toBeVisible();
}

/** Dispatch a cancelable touchmove on `selector` and report whether the page may still react to it. */
const touchMoveIsPrevented = (page: Page, selector: string) =>
  page.evaluate((sel) => {
    const el = document.querySelector(sel)!;
    // Desktop WebKit has no Touch constructor; the guard listens for the event name, so a plain event does.
    const ev = new Event("touchmove", { cancelable: true, bubbles: true });
    el.dispatchEvent(ev);
    return ev.defaultPrevented;
  }, selector);

test("the document never rubber-bands or pulls to refresh", async ({
  page,
}) => {
  await open(page);
  const behavior = await page.evaluate(
    () => getComputedStyle(document.documentElement).overscrollBehaviorY,
  );
  expect(behavior).toBe("none");
});

test("a drag on the panel's grip or header, or on the scrim, cannot move the page", async ({
  page,
}) => {
  await open(page);
  expect(
    await touchMoveIsPrevented(
      page,
      '[data-yunui="status-island"] [data-island-drag]',
    ),
  ).toBe(true);
  expect(
    await touchMoveIsPrevented(page, '[data-yunui="status-island-scrim"]'),
  ).toBe(true);
});

test("the page scroller keeps its position and the body is pinned while the panel is open", async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") return route.fulfill({ json: status });
    return route.fulfill({ status: 404, json: { detail: "Not Found" } });
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  const scroller = page.getByTestId("page-scroll");
  await scroller.evaluate((e) => {
    e.scrollTop = 300;
  });
  const before = await scroller.evaluate((e) => e.scrollTop);
  await page.getByTestId("footer-compact").click();
  await expect(page.getByRole("dialog", { name: "引擎狀態" })).toBeVisible();
  expect(await page.evaluate(() => document.body.style.position)).toBe("fixed");
  await page.evaluate(() => {
    const el = document.querySelector('[data-yunui="status-island-scrim"]')!;
    el.dispatchEvent(
      new Event("touchmove", { cancelable: true, bubbles: true }),
    );
  });
  expect(await scroller.evaluate((e) => e.scrollTop)).toBe(before);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog", { name: "引擎狀態" })).toBeHidden();
  expect(await page.evaluate(() => document.body.style.position)).toBe("");
  expect(await scroller.evaluate((e) => e.scrollTop)).toBe(before);
});

test("the panel's own scroll area contains its overscroll", async ({
  page,
}) => {
  await open(page);
  const contain = await page.evaluate(() => {
    const el = document.querySelector(
      '[data-yunui="status-island"] [data-yunui="scroll-fade"]',
    )!;
    return getComputedStyle(el).overscrollBehaviorY;
  });
  expect(contain).toBe("contain");
});

test("a new page opens at its top, not at the previous page's scroll offset", async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") return route.fulfill({ json: status });
    return route.fulfill({ status: 404, json: { detail: "Not Found" } });
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  const scroller = page.getByTestId("page-scroll");
  await expect(page.getByTestId("overview")).toBeVisible();
  await scroller.evaluate((e) => {
    e.style.minHeight = "0";
    const filler = document.createElement("div");
    filler.style.height = "3000px";
    filler.dataset.filler = "1";
    e.appendChild(filler);
    e.scrollTop = 800;
  });
  expect(await scroller.evaluate((e) => e.scrollTop)).toBeGreaterThan(500);
  await page.evaluate(() => {
    location.hash = "#/diagnostics";
  });
  await expect(
    page.getByRole("heading", { level: 1, name: "引擎診斷" }),
  ).toBeVisible();
  // The filler is still there, so a clamped offset cannot hide the bug: the page itself must have reset it.
  await expect.poll(() => scroller.evaluate((e) => e.scrollTop)).toBe(0);
});
