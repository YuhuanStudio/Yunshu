import { expect, test, type Page, type Route } from "@playwright/test";

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });

const status = {
  object: "yunshu.status",
  version: "t-1",
  state: "running",
  uptime_s: 90,
  load_error: null,
  models: [
    {
      id: "org/qwen-mlx",
      type: "LLM",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 5,
      idle_s: 0,
      keep_alive_s: null,
      expires_in_s: null,
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
    mean_decode_tps: 42,
  },
};

async function install(page: Page, opts: { config?: "ok" | 404 } = {}) {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: async () => undefined },
    });
  });
  await page.route("**/v1/yunshu/status", (route) => json(route, status));
  await page.route("**/v1/models", (route) =>
    json(route, { object: "list", data: [] }),
  );
  await page.route("**/debug/**", (route) =>
    json(route, { detail: "Not Found" }, 404),
  );
  await page.route("**/openapi.json", (route) =>
    json(route, {
      paths: { "/v1/models": { get: { summary: "List", tags: ["x"] } } },
    }),
  );
  await page.route("**/v1/yunshu/config*", (route) =>
    opts.config === 404
      ? json(route, { detail: "Not Found" }, 404)
      : json(route, {
          object: "yunshu.config",
          include: "stable",
          settings: [
            {
              name: "YUNSHU_PORT",
              value: 9000,
              default: 8000,
              source: "env",
              stability: "stable",
              category: "server",
              description: "Port",
            },
            {
              name: "YUNSHU_LOG_LEVEL",
              value: "INFO",
              default: "INFO",
              source: "default",
              stability: "stable",
              category: "observability",
              description: "Level",
            },
            {
              name: "YUNSHU_AUTH_TOKEN",
              value: "***",
              default: null,
              source: "env",
              stability: "stable",
              category: "auth",
              description: "Token",
            },
          ],
          warnings: [],
          experimental_count: 1,
          experimental_max: 8,
        }),
  );
}

test("diagnostics without /debug shows one explanation, no empty tiles", async ({
  page,
}) => {
  await install(page);
  await page.goto("/console/#/diagnostics", { waitUntil: "domcontentloaded" });
  const note = page.getByTestId("debug-disabled");
  await expect(note).toContainText("YUNSHU_DEBUG_ROUTES");
  const body = page.getByTestId("diagnostics");
  await expect(body).not.toContainText("Not Found");
  await expect(body).not.toContainText("個邏輯核心");
  await expect(body.getByText("—%")).toHaveCount(0);
  await expect(body.getByText("未啟用").first()).toBeVisible();
  await page.screenshot({
    path: "test-results/round4-d-diagnostics-off.png",
    fullPage: true,
  });
});

test("settings lists effective config with changed rows and masked secrets", async ({
  page,
}) => {
  await install(page);
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  const view = page.getByTestId("config-view");
  await expect(view.getByText("YUNSHU_PORT")).toBeVisible();
  await expect(view.locator('tr[data-changed="true"]')).toHaveCount(2);
  await expect(view.getByText("***").first()).toBeVisible();
  await view.getByLabel("搜尋設定").fill("log");
  await expect(view.getByText("YUNSHU_PORT")).toHaveCount(0);
  await expect(view.getByText("YUNSHU_LOG_LEVEL")).toBeVisible();
  await view.scrollIntoViewIfNeeded();
  await page.screenshot({
    path: "test-results/round4-d-config.png",
    fullPage: true,
  });
});

test("settings config falls back calmly on 404", async ({ page }) => {
  await install(page, { config: 404 });
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("config-view")).toContainText(
    "尚未提供有效設定",
  );
});

test("api catalog rows have a copy button", async ({ page }) => {
  await install(page);
  await page.goto("/console/#/api", { waitUntil: "domcontentloaded" });
  await expect(
    page.getByRole("button", { name: /複製 GET \/v1\/models/ }),
  ).toBeVisible();
});

test("playground: 5xx is zh-TW with details, clearing offers 復原, reasoning folds", async ({
  page,
}) => {
  await install(page);
  let fail = true;
  await page.route("**/v1/chat/completions", (route) => {
    if (fail) return json(route, { detail: "kaboom engine exploded" }, 500);
    const ev = (o: unknown) => `data: ${JSON.stringify(o)}\n\n`;
    return route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body:
        ev({ choices: [{ delta: { reasoning_content: "hmm thinking" } }] }) +
        ev({ choices: [{ delta: { content: "final answer" } }] }) +
        ev({ choices: [{ delta: {}, finish_reason: "stop" }] }) +
        ev({
          choices: [],
          usage: { prompt_tokens: 3, completion_tokens: 4 },
        }) +
        "data: [DONE]\n\n",
    });
  });
  await page.goto("/console/#/playground", { waitUntil: "domcontentloaded" });
  const pg = page.getByTestId("playground");
  const box = pg.getByPlaceholder("輸入測試提示詞…");
  await box.fill("hi");
  await pg.getByRole("button", { name: "傳送測試" }).click();
  await expect(pg).toContainText("引擎內部錯誤");
  await expect(pg).not.toContainText("OpenAI API request failed");
  await pg.getByText("詳細資訊").click();
  await expect(pg).toContainText("kaboom engine exploded");
  fail = false;
  await box.fill("again");
  await pg.getByRole("button", { name: "傳送測試" }).click();
  await expect(pg).toContainText("final answer");
  const think = pg.getByRole("button", { name: /思考過程/ });
  await expect(think).toHaveAttribute("aria-expanded", "false");
  await think.click();
  await expect(think).toHaveAttribute("aria-expanded", "true");
  await pg.getByRole("button", { name: "新測試" }).click();
  await expect(pg).not.toContainText("final answer");
  // 復原 is a toast action now (rendered by the app Toaster, outside the playground).
  await expect(page.getByText("已清除這段測試")).toBeVisible();
  await page.getByRole("button", { name: "復原" }).click();
  await expect(pg).toContainText("final answer");
});
