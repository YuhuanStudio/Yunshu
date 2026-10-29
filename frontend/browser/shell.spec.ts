import { expect, test, type Page } from "@playwright/test";

async function install(page: Page) {
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    const ok = path === "/v1/yunshu/status";
    return route.fulfill({
      status: ok ? 200 : 404,
      contentType: "application/json",
      body: JSON.stringify(
        ok
          ? {
              object: "yunshu.status",
              version: "fixture-1.0",
              state: "ready",
              uptime_s: 100,
              load_error: null,
              models: [],
              memory: { active_gb: 12, cache_gb: 2, peak_gb: 14, total_gb: 64 },
              requests: {
                active: 0,
                queued: 0,
                prefill: 0,
                decode: 0,
                items: [],
              },
              last: null,
              throughput: {
                window_s: 60,
                requests: 0,
                prompt_tokens: 0,
                completion_tokens: 0,
                live_decode_tps: null,
                mean_prefill_tps: 800,
                mean_decode_tps: 35,
              },
            }
          : { detail: "fixture endpoint not found" },
      ),
    });
  });
}

const geometry = (page: Page) =>
  page.evaluate(() => {
    // Scroll every scroll container (and the document) to its end.
    for (const e of document.querySelectorAll("*"))
      if (
        e.scrollHeight > e.clientHeight + 1 &&
        /auto|scroll/.test(getComputedStyle(e).overflowY)
      )
        e.scrollTop = 1e6;
    window.scrollTo(0, 1e6);
    const bar = document.querySelector("ul[aria-label='引擎狀態']");
    const doc = document.scrollingElement!;
    return {
      barBottom: bar?.getBoundingClientRect().bottom ?? null,
      inner: window.innerHeight,
      docScroll: doc.scrollHeight,
      scrollY: window.scrollY,
    };
  });

test.describe("shell scroll containment", () => {
  test("long page: status band stays pinned to the viewport bottom and the document never scrolls", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 640 });
    await install(page);
    await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("overview")).toBeVisible();
    await page.waitForTimeout(1500);
    const g = await geometry(page);
    expect(g.barBottom).not.toBeNull();
    // the status islands float a small gap above the viewport bottom
    expect(g.barBottom!).toBeLessThanOrEqual(g.inner);
    expect(g.barBottom!).toBeGreaterThanOrEqual(g.inner - 20);
    expect(g.scrollY).toBe(0);
    expect(g.docScroll).toBeLessThanOrEqual(g.inner);
  });

  test("short page: no blank scroll height and the band sits at the viewport bottom", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 1800 });
    await install(page);
    await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("requests")).toBeVisible();
    await page.waitForTimeout(1500);
    const g = await geometry(page);
    expect(g.barBottom!).toBeLessThanOrEqual(g.inner);
    expect(g.barBottom!).toBeGreaterThanOrEqual(g.inner - 20);
    expect(g.docScroll).toBeLessThanOrEqual(g.inner);
  });
});

test.describe("status band never covers content", () => {
  for (const route of [
    "overview",
    "requests",
    "models",
    "diagnostics",
    "api",
    "settings",
    "playground",
  ]) {
    test(`${route}: scrolled to the end, the last content ends above the band`, async ({
      page,
    }) => {
      await page.setViewportSize({ width: 1280, height: 560 });
      await install(page);
      await page.goto(`/console/#/${route}`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(1500);
      const r = await page.evaluate(() => {
        // The islands float over the page bottom (no reserved row): scrolled to the end, whatever lies
        // horizontally under them must end above their top edge.
        const band = document
          .querySelector("ul[aria-label='引擎狀態']")!
          .getBoundingClientRect();
        const main = document.querySelector("main")!;
        for (const e of [main, ...main.querySelectorAll("*")])
          if (
            e.scrollHeight > e.clientHeight + 1 &&
            /auto|scroll/.test(getComputedStyle(e).overflowY)
          )
            e.scrollTop = 1e6;
        let lowest = 0;
        for (const el of main.querySelectorAll("*")) {
          const b = el.getBoundingClientRect();
          if (
            b.height > 0 &&
            b.width > 0 &&
            b.left < band.right &&
            b.right > band.left &&
            b.top < band.bottom &&
            getComputedStyle(el).visibility !== "hidden" &&
            el.children.length === 0
          )
            lowest = Math.max(lowest, b.bottom);
        }
        return { top: band.top, lowest };
      });
      expect(r.lowest).toBeLessThanOrEqual(r.top + 0.5);
    });
  }
});

test("a page whose code fails to load once (engine restarting) recovers on its own", async ({
  page,
}) => {
  // The console is served by the engine: a page chunk requested while it restarts fails once.
  await install(page);
  let blocked = 0;
  await page.route(/\/(src\/Settings\.tsx|assets\/Settings-[^/]*\.js)(\?|$)/, (route) => {
    if (blocked++ === 0) return route.abort("connectionrefused");
    return route.fallback();
  });
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("page-error")).toHaveCount(0, { timeout: 15_000 });
  await expect(page.getByTestId("settings")).toBeVisible({ timeout: 15_000 });
  expect(blocked).toBeGreaterThan(1);
});

test("a page whose code never loads reloads once, then shows its error instead of looping", async ({
  page,
}) => {
  await install(page);
  await page.route(/\/(src\/Settings\.tsx|assets\/Settings-[^/]*\.js)(\?|$)/, (route) =>
    route.abort("connectionrefused"),
  );
  let loads = 0;
  page.on("load", () => loads++);
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("page-error")).toBeVisible({ timeout: 20_000 });
  await page.waitForTimeout(2000);
  expect(loads).toBe(2); // the first load and one automatic reload, no more
  await expect(page.getByTestId("page-error")).toBeVisible();
});
