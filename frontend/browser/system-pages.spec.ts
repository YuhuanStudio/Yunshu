import { expect, test, type Page, type Route } from "@playwright/test";

type Win = { __copied: string[] };

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });

async function install(page: Page, debug: boolean) {
  await page.addInitScript(() => {
    (window as unknown as Win).__copied = [];
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text: string) => {
          (window as unknown as Win).__copied.push(text);
        },
      },
    });
  });
  await page.route("**/v1/yunshu/status", (route) =>
    json(route, {
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
    }),
  );
  await page.route("**/debug/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (!debug) return json(route, { detail: "off" }, 404);
    return json(
      route,
      path === "/debug/system"
        ? {
            cpu: { percent: 12, logical_cores: 16 },
            memory: { percent: 40, used_bytes: 2e10 },
            gpu: { active_bytes: 1.2e10 },
          }
        : {},
    );
  });
  await page.route("**/openapi.json", (route) => json(route, { paths: {} }));
}

test("diagnostics lists health checks, readouts and copies the bundle", async ({
  page,
}) => {
  await install(page, true);
  await page.goto("/console/#/diagnostics", { waitUntil: "domcontentloaded" });
  const health = page.getByTestId("health-checks");
  await expect(health.getByText("引擎狀態")).toBeVisible();
  await expect(health.getByText("診斷介面")).toBeVisible();
  await expect(health.getByText("可用", { exact: true })).toBeVisible();
  await expect(page.getByTestId("resource-readouts")).toContainText("16");
  await page.getByRole("button", { name: "複製診斷資料" }).click();
  await expect(page.getByRole("button", { name: "已複製" })).toBeVisible();
  const copied = await page.evaluate(
    () => (window as unknown as Win).__copied[0],
  );
  const bundle = JSON.parse(copied);
  expect(bundle.status.version).toBe("t-1");
  expect(bundle.debug_system.cpu.logical_cores).toBe(16);
});

test("api access generates commands for the base URL and model", async ({
  page,
}) => {
  await install(page, false);
  await page.goto("/console/#/api", { waitUntil: "domcontentloaded" });
  const origin = new URL(page.url()).origin;
  for (const id of [
    "claude-code",
    "codex",
    "opencode",
    "openai",
    "anthropic",
    "curl",
  ]) {
    const card = page.getByTestId(`integration-${id}`);
    await expect(card).toBeVisible();
    // opencode.json is long and collapsed by CodeBlock; the unit test covers its URL.
    if (id !== "opencode") await expect(card).toContainText(origin);
    await expect(card).toContainText("org/qwen-mlx");
    await expect(
      card.getByRole("button", { name: /複製/ }).first(),
    ).toBeVisible();
  }
  await expect(page.getByTestId("api-catalog")).toBeVisible();
  await expect(page.getByRole("textbox", { name: "搜尋 API" })).toBeVisible();
});

test("settings groups connection, appearance, retention and shortcuts", async ({
  page,
}) => {
  await install(page, false);
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  const settings = page.getByTestId("settings");
  await expect(settings.getByLabel("存取權杖")).toBeVisible();
  await expect(
    settings.getByRole("button", { name: "儲存並連線" }),
  ).toBeVisible();
  await expect(
    settings.getByRole("switch", { name: "深色介面" }),
  ).toBeVisible();
  await expect(
    settings.getByRole("combobox", { name: "閒置保留時間" }),
  ).toBeVisible();
  await expect(settings.getByText("開啟命令面板")).toBeVisible();
});
