import { expect, test, type Page } from "@playwright/test";

const model = (id: string, over: Record<string, unknown> = {}) => ({
  id,
  type: "text",
  loaded: false,
  loading: false,
  pinned: false,
  expires_in_s: null,
  ...over,
});

const status = (models: unknown[]) => ({
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "ready",
  uptime_s: 100,
  load_error: null,
  models,
  memory: {
    active_gb: 20,
    cache_gb: 1,
    peak_gb: 30,
    total_gb: 64,
    pressure: 0.3,
  },
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
});

const ledger = (over: Record<string, unknown> = {}) => ({
  object: "yunshu.memory",
  total_gb: 64,
  free_gb: 20,
  host: { pressure_level: "normal", swap_used_gb: 0.5 },
  mlx: {
    active_gb: 30,
    cache_gb: 2,
    peak_gb: 40,
    recommended_working_set_gb: 48,
  },
  owners: [
    {
      kind: "weights",
      id: "org/big",
      bytes: 1,
      gb: 25,
      reclaimable: true,
      estimated: false,
      source: "model parameters",
    },
    {
      kind: "apc_ram",
      id: "org/big",
      bytes: 1,
      gb: 2,
      reclaimable: true,
      estimated: false,
      source: "x",
    },
    {
      kind: "live_kv",
      id: null,
      bytes: null,
      gb: null,
      reclaimable: false,
      estimated: false,
      source: null,
    },
    {
      kind: "other",
      id: null,
      bytes: 1,
      gb: 3,
      reclaimable: false,
      estimated: true,
      source: "residual",
    },
  ],
  attribution_overshoot_gb: null,
  limits: { apc_max_gb: null, apc_warm_max_gb: null, guard_margin_pct: null },
  ...over,
});

async function install(page: Page, memory: unknown | null) {
  const models = [
    model("org/big", { loaded: true, size_gb: 25, expires_in_s: 600 }),
    model("org/small-no-size"),
    model("org/huge", { size_gb: 40 }),
    model("org/fits", { size_gb: 5 }),
  ];
  await page.route("**/v1/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    const json = (code: number, body: unknown) =>
      route.fulfill({
        status: code,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    if (path === "/v1/yunshu/status") return json(200, status(models));
    if (path === "/v1/yunshu/memory")
      return memory == null
        ? json(404, { detail: "Not Found" })
        : json(200, memory);
    if (path.startsWith("/v1/models/")) return json(200, { id: "card" });
    return json(404, { detail: "fixture" });
  });
}

test.describe("models page, mobile and memory ledger", () => {
  test("every action is reachable at 390, 768 and 1024 px", async ({
    page,
  }) => {
    await install(page, null);
    for (const width of [390, 768, 1024]) {
      await page.setViewportSize({ width, height: 900 });
      await page.goto("/console/#/models", { waitUntil: "domcontentloaded" });
      const models = page.getByTestId("models");
      for (const name of ["測試", "預熱", "卸載"]) {
        const button = models
          .getByRole("button", { name, exact: true })
          .first();
        await expect(button).toBeVisible();
        const box = await button.boundingBox();
        expect(box!.x).toBeGreaterThanOrEqual(0);
        expect(box!.x + box!.width).toBeLessThanOrEqual(width);
      }
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - window.innerWidth,
      );
      expect(overflow).toBeLessThanOrEqual(0);
    }
  });

  test("unknown size keeps the dash and explains why", async ({ page }) => {
    await install(page, null);
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/console/#/models", { waitUntil: "domcontentloaded" });
    const dash = page.getByTestId("size-unknown").first();
    await expect(dash).toHaveText("—");
    await expect(dash).toHaveAttribute("title", /沒有回報模型大小/);
  });

  test("fit-before-load informs without blocking", async ({ page }) => {
    await install(page, ledger());
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/console/#/models", { waitUntil: "domcontentloaded" });
    await expect(
      page.locator('[data-testid="fit-hint"][data-verdict="no"]:visible'),
    ).toContainText("可能放不下");
    await expect(
      page.locator('[data-testid="fit-hint"][data-verdict="fits"]:visible'),
    ).toContainText("放得下");
    const row = page.getByRole("row").filter({ hasText: "huge" });
    await expect(
      row.getByRole("button", { name: "載入", exact: true }),
    ).toBeEnabled();
  });

  test("the summary card and every row use the same available-memory figure", async ({
    page,
  }) => {
    await install(
      page,
      ledger({ free_gb: 93.1, free_bytes: Math.round(93.1 * 2 ** 30) }),
    );
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/console/#/models", { waitUntil: "domcontentloaded" });
    const card = page.getByTestId("memory-free");
    await expect(card).toContainText("系統可用 93.1 GB");
    await expect(card).toHaveAttribute("data-source", "system");
    // Every sized, unloaded row measures itself against the very same number.
    const hints = page.locator('[data-testid="fit-hint"]:visible');
    await expect(hints.first()).toContainText("系統可用 93.1 GB");
    for (const text of await hints.allTextContents())
      expect(text).toContain("系統可用 93.1 GB");
  });

  test("without a ledger the card says it is Metal's remainder, not the system figure", async ({
    page,
  }) => {
    await install(page, null);
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/console/#/models", { waitUntil: "domcontentloaded" });
    const card = page.getByTestId("memory-free");
    await expect(card).toHaveAttribute("data-source", "metal");
    await expect(card).toContainText("Metal 之外");
  });

  test("detail page renders the ledger with estimated and unknown values", async ({
    page,
  }) => {
    await install(page, ledger());
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/console/#/models/org%2Fbig", {
      waitUntil: "domcontentloaded",
    });
    const l = page.getByTestId("memory-ledger");
    await expect(l).toBeVisible();
    const table = l.getByRole("table", { name: "記憶體持有者" });
    await expect(
      table.getByRole("row").filter({ hasText: "進行中請求的 KV" }),
    ).toContainText("未知");
    await expect(
      table.getByRole("row").filter({ hasText: "其他" }),
    ).toContainText("估算");
    await expect(
      table.getByRole("row").filter({ hasText: "模型權重" }),
    ).not.toContainText("估算");
    await expect(
      l.getByRole("img", { name: "Metal 記憶體依持有者分佈" }),
    ).toBeVisible();
  });

  test("capacity gauge turns to the danger tone past 92 percent", async ({
    page,
  }) => {
    await install(
      page,
      ledger({
        mlx: {
          active_gb: 62,
          cache_gb: 1,
          peak_gb: 63,
          recommended_working_set_gb: 48,
        },
      }),
    );
    await page.goto("/console/#/models/org%2Fbig", {
      waitUntil: "domcontentloaded",
    });
    await expect(
      page.getByTestId("memory-ledger").getByText("接近上限"),
    ).toBeVisible();
  });

  test("an older server without the ledger gets a quiet fallback", async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await install(page, null);
    await page.goto("/console/#/models/org%2Fbig", {
      waitUntil: "domcontentloaded",
    });
    await expect(page.getByTestId("memory-ledger-unsupported")).toBeVisible();
    await expect(page.getByTestId("memory-ledger")).toHaveCount(0);
    expect(errors).toEqual([]);
  });
});
