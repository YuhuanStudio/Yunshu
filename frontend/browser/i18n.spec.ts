import { expect, test, type Page } from "@playwright/test";

const items = [
  {
    request_id: "i18n-queue-01",
    elapsed_s: 6.2,
    phase: "queued",
    model: "org/qwen-mlx",
    queue_position: 2,
    queue_est_wait_ms: 1800,
  },
  {
    request_id: "i18n-prefill-01",
    elapsed_s: 3.4,
    phase: "prefill",
    model: "org/qwen-mlx",
    prompt_tokens: 4000,
    cached_tokens: 1000,
    processed_tokens: 2500,
    percent: 62.5,
    eta_s: 1.2,
    tokens_per_second: 900,
  },
  {
    request_id: "i18n-decode-01",
    elapsed_s: 9,
    phase: "decode",
    model: "org/qwen-mlx",
    prompt_tokens: 600,
    cached_tokens: 200,
    completion_tokens: 42,
    tokens_per_second: 28.5,
  },
];
const recent = Array.from({ length: 4 }, (_, i) => ({
  request_id: `ring-req-0123456789abcdef-${i}`,
  t: 1_800_000_000 - i,
  path: "/v1/chat/completions",
  model: "org/qwen-mlx",
  status: 200,
  finish_reason: "stop",
  stream: true,
  t0_wall: 1_799_999_990 - i,
  offsets_ms: {
    arrive: 0,
    admit: 40,
    first_token: 340,
    last_token: 2340,
    done: 2360,
  },
  queue_wait_ms: 40,
  ttft_ms: 300,
  prompt_tokens: 4000,
  cached_tokens: 1000,
  completion_tokens: 80,
  prefill_tps: 800,
  decode_tps: 40,
  cache: { tier: "ram", reload_ms: null },
  speculative: { mode: "mtp", acceptance_rate: 0.8, rounds: 20 },
  cancelled: false,
}));

async function install(page: Page) {
  let n = 0;
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    const json = (status: number, body: unknown) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    if (path === "/v1/yunshu/status") {
      n += 1;
      return json(200, {
        object: "yunshu.status",
        version: "fixture-1.0",
        state: "ready",
        uptime_s: 3600 + n,
        load_error: null,
        models: [
          {
            id: "org/qwen-mlx",
            type: "llm",
            loaded: true,
            size_gb: 12.3,
            keep_alive_s: 300,
            idle_s: 4,
            expires_in_s: 296,
          },
        ],
        memory: { active_gb: 12, cache_gb: 2, peak_gb: 14, total_gb: 64 },
        requests: { active: 2, queued: 1, prefill: 1, decode: 1, items },
        last: null,
        throughput: {
          window_s: 60,
          requests: 3,
          prompt_tokens: 100,
          completion_tokens: 50,
          live_decode_tps: 28.5,
          mean_prefill_tps: 800,
          mean_decode_tps: 35,
        },
      });
    }
    if (path === "/v1/yunshu/requests/recent")
      return json(200, { object: "list", data: recent });
    return json(404, { detail: "fixture endpoint not found" });
  });
}

const PAGES = [
  "overview",
  "requests",
  "models",
  "playground",
  "diagnostics",
  "api",
  "settings",
];
const LOCALES = [
  { locale: "zh-TW", lang: "zh-Hant-TW", overview: "引擎總覽" },
  { locale: "zh-CN", lang: "zh-Hans-CN", overview: "引擎总览" },
  { locale: "en", lang: "en", overview: "Overview" },
] as const;

for (const l of LOCALES) {
  test.describe(`locale ${l.locale}`, () => {
    test.use({ locale: l.locale });
    test("every page renders with no missing-key marker and the right <html lang>", async ({
      page,
    }) => {
      await install(page);
      for (const p of PAGES) {
        await page.goto(`/console/#/${p}`, { waitUntil: "domcontentloaded" });
        await expect(page.locator("main")).toBeVisible();
        await page.waitForTimeout(500);
        expect(await page.locator("html").getAttribute("lang")).toBe(l.lang);
        const text = await page.locator("body").innerText();
        expect(text, `page ${p}`).not.toContain("⟦");
        if (p === "overview")
          await expect(
            page.getByRole("link", { name: l.overview }).first(),
          ).toBeVisible();
        if (l.locale === "en")
          expect(text, `page ${p}`).not.toMatch(/[㐀-鿿]{2}/);
      }
    });
  });
}

test.describe("switching", () => {
  test.use({ locale: "zh-TW" });
  test("switching updates the UI and <html lang> without a reload and persists", async ({
    page,
  }) => {
    await install(page);
    await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("overview")).toBeVisible();
    await page.evaluate(
      () => ((window as unknown as Record<string, unknown>).__noReload = 1),
    );
    const switcher = page.getByRole("button", { name: "語言" }).first();
    await switcher.click();
    await page
      .getByRole("option", { name: "English" })
      .or(page.getByText("English", { exact: true }))
      .first()
      .click();
    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    await expect(
      page.getByRole("link", { name: "Overview" }).first(),
    ).toBeVisible();
    expect(
      await page.evaluate(
        () => (window as unknown as Record<string, unknown>).__noReload,
      ),
    ).toBe(1);
    expect(
      await page.evaluate(() => localStorage.getItem("yunshu.console.locale")),
    ).toBe("en");
    await page.reload({ waitUntil: "domcontentloaded" });
    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    expect(await page.locator("body").innerText()).not.toContain("⟦");
  });

  test("Settings has a language option that switches to 简体中文", async ({
    page,
  }) => {
    await install(page);
    await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("settings")).toBeVisible();
    await page
      .getByTestId("settings")
      .getByRole("button", { name: "語言" })
      .click();
    await page.getByText("简体中文", { exact: true }).last().click();
    await expect(page.locator("html")).toHaveAttribute("lang", "zh-Hans-CN");
    await expect(page.getByTestId("settings")).toContainText("引擎连接");
  });
});

for (const width of [1024, 390]) {
  test.describe(`English layout at ${width}px`, () => {
    test.use({ locale: "en" });
    test("no page overflows horizontally", async ({ page }) => {
      await page.setViewportSize({ width, height: 800 });
      await install(page);
      for (const p of PAGES) {
        await page.goto(`/console/#/${p}`, { waitUntil: "domcontentloaded" });
        await expect(page.locator("main")).toBeVisible();
        await page.waitForTimeout(500);
        const over = await page.evaluate(() => {
          const doc = document.documentElement;
          const bad: string[] = [];
          for (const e of document.querySelectorAll(
            "main *, header *, footer *",
          )) {
            const r = e.getBoundingClientRect();
            if (
              r.width > 0 &&
              r.right > window.innerWidth + 1 &&
              !e.closest("[aria-hidden='true'], .sr-only, [inert]")
            ) {
              const sc = e.closest(
                "[class*='overflow-x'], [class*='overflow-auto'], pre, table",
              );
              if (!sc)
                bad.push(
                  `${e.tagName}.${String(e.className).slice(0, 40)} right=${Math.round(r.right)}`,
                );
            }
          }
          return {
            scroll: doc.scrollWidth - window.innerWidth,
            bad: bad.slice(0, 5),
          };
        });
        expect(over.scroll, `page ${p} document scroll`).toBeLessThanOrEqual(1);
        expect(over.bad, `page ${p} elements past the viewport`).toEqual([]);
      }
    });
  });
}
