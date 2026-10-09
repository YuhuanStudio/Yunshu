import { expect, test, type Page, type Route } from "@playwright/test";

type World = {
  uptime: number;
  fail: boolean;
  needToken: boolean;
  memory: { active_gb: number; total_gb: number };
  queued: number;
  oldestWait: number;
  models: {
    id: string;
    type: string;
    loaded: boolean;
    loading: boolean;
    pinned: boolean;
    size_gb: number;
  }[];
  loadCalls: string[];
  chatPending: boolean;
};
const world = (over: Partial<World> = {}): World => ({
  uptime: 5000,
  fail: false,
  needToken: false,
  memory: { active_gb: 30, total_gb: 137 },
  queued: 0,
  oldestWait: 0,
  models: [
    {
      id: "Qwen3.8-27B",
      type: "LLM",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 17.6,
    },
  ],
  loadCalls: [],
  chatPending: false,
  ...over,
});

async function install(page: Page, w: World, { configured = true } = {}) {
  if (configured)
    await page.addInitScript(() =>
      localStorage.setItem("yunshu.console.url", location.origin),
    );
  await page.route("**/v1/**", async (route: Route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    const json = (status: number, body: unknown) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    if (w.fail) return route.abort("connectionrefused");
    if (w.needToken && req.headers()["authorization"] !== "Bearer secret")
      return json(401, { detail: "unauthorized" });
    if (path === "/v1/yunshu/status")
      return json(200, {
        object: "yunshu.status",
        version: "fixture",
        state: "running",
        uptime_s: w.uptime,
        load_error: null,
        models: w.models,
        memory: { ...w.memory, cache_gb: 1, peak_gb: w.memory.active_gb },
        requests: {
          active: w.queued,
          queued: w.queued,
          prefill: 0,
          decode: 0,
          items: Array.from({ length: w.queued }, (_, i) => ({
            request_id: `q${i}`,
            elapsed_s: i === 0 ? w.oldestWait : 1,
            phase: "queued",
            model: "Qwen3.8-27B",
          })),
        },
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
    if (path === "/v1/models/load") {
      w.loadCalls.push(JSON.parse(req.postData() ?? "{}").model);
      return json(200, { status: "loaded" });
    }
    if (path === "/v1/chat/completions") {
      w.chatPending = true;
      await new Promise(() => undefined); // never answers: the console must abort it itself
    }
    return json(404, { detail: "fixture endpoint not found" });
  });
}

test("health verdict: healthy, then attention with reasons worst first", async ({
  page,
}) => {
  const w = world();
  await install(page, w);
  await page.goto("/console/#/overview");
  const verdict = page.getByTestId("health-verdict");
  await expect(verdict).toHaveAttribute("data-level", "ok");
  await expect(verdict).toContainText("健康");
  w.memory = { active_gb: 125, total_gb: 137 };
  w.queued = 4;
  w.oldestWait = 43;
  await expect(verdict).toHaveAttribute("data-level", "bad", { timeout: 8000 });
  await expect(verdict).toContainText("異常");
  await expect(verdict).toContainText("記憶體 91%");
  await expect(verdict).toContainText("排隊");
});

test("a refused token keeps the overview and asks for the token in place", async ({
  page,
}) => {
  const w = world({ needToken: true });
  await install(page, w);
  await page.goto("/console/#/overview");
  await expect(page.getByRole("heading", { name: "引擎總覽" })).toBeVisible();
  await expect(page.getByText("連接本機推理引擎")).toHaveCount(0);
  const field = page.getByLabel("解鎖金鑰").first();
  await expect(field).toBeVisible();
  await field.fill("secret");
  await page.getByRole("button", { name: "解鎖" }).click();
  await expect(page.getByTestId("health-verdict")).toHaveAttribute(
    "data-level",
    "ok",
    { timeout: 8000 },
  );
  await expect(page.getByRole("button", { name: "解鎖" })).toHaveCount(0);
});

test("a restart raises a toast and a bell entry; the bell is calm until then", async ({
  page,
}) => {
  const w = world();
  await install(page, w);
  await page.goto("/console/#/overview");
  await expect(page.getByTestId("health-verdict")).toHaveAttribute(
    "data-level",
    "ok",
  );
  await expect(page.getByTestId("bell-badge")).toBeHidden();
  w.uptime = 12;
  await expect(
    page.getByText(/引擎已重新啟動 · \d\d:\d\d/).first(),
  ).toBeVisible({ timeout: 9000 });
  await expect(page.getByTestId("bell-badge")).toBeVisible();
  await page.getByRole("button", { name: /通知，1 則未讀/ }).click();
  await expect(
    page
      .getByRole("dialog")
      .or(page.locator("[data-radix-popper-content-wrapper]"))
      .getByText(/引擎已重新啟動/),
  ).toBeVisible();
});

test("offline: panels dim and say where the data stopped", async ({ page }) => {
  const w = world();
  await install(page, w);
  await page.goto("/console/#/overview");
  await expect(page.getByTestId("health-verdict")).toHaveAttribute(
    "data-level",
    "ok",
  );
  w.fail = true;
  await expect(page.getByTestId("stale-stamp")).toContainText(
    /資料停在 \d\d:\d\d/,
    { timeout: 20000 },
  );
  await expect(page.getByTestId("overview-stats")).toHaveClass(/opacity-60/);
  await expect(page.getByTestId("health-verdict")).toHaveAttribute(
    "data-level",
    "bad",
  );
  await expect(page).toHaveTitle(/離線/);
});

test("palette verbs deep-link into the page that owns the action", async ({
  page,
}) => {
  await install(page, world());
  await page.goto("/console/#/overview");
  await page.keyboard.press("Control+k");
  await page
    .getByRole("combobox")
    .or(page.getByRole("textbox"))
    .first()
    .fill("清除快取");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#\/cache\?action=clear/);
  await page.keyboard.press("Control+k");
  await page
    .getByRole("combobox")
    .or(page.getByRole("textbox"))
    .first()
    .fill("前往日誌");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#\/logs$/);
  await page.keyboard.press("Control+k");
  await page
    .getByRole("combobox")
    .or(page.getByRole("textbox"))
    .first()
    .fill("切換語言");
  await expect(page.getByText("切換語言 · English")).toBeVisible();
});

test("after navigation focus is on the page heading", async ({ page }) => {
  await install(page, world());
  await page.goto("/console/#/overview");
  await page
    .getByRole("link", { name: "請求與效能" })
    .first()
    .click()
    .catch(() => undefined);
  await page.evaluate(() => (location.hash = "#/diagnostics"));
  await expect
    .poll(() => page.evaluate(() => document.activeElement?.tagName))
    .toBe("H1");
});

test("? opens the shortcuts sheet and g then r goes to requests", async ({
  page,
}) => {
  await install(page, world());
  await page.goto("/console/#/overview");
  await page.locator("body").click();
  await page.keyboard.press("?");
  await expect(
    page.getByRole("dialog").getByText("鍵盤快捷鍵").first(),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await page.keyboard.press("g");
  await page.keyboard.press("r");
  await expect(page).toHaveURL(/#\/requests/);
});

test("the sidebar has the four groups; downloads, cache, logs and keys are tabs", async ({
  page,
}) => {
  await install(page, world());
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/console/#/overview");
  const nav = page.getByRole("navigation", { name: "控制台導覽" });
  for (const group of ["監控", "模型", "開發", "管理"])
    await expect(nav.getByText(group, { exact: true })).toBeVisible();
  for (const [from, label, hash] of [
    ["models", "下載", "downloads"],
    ["models", "快取", "cache"],
    ["diagnostics", "日誌", "logs"],
    ["settings", "金鑰", "keys"],
    ["settings", "API 接入", "api"],
  ] as const) {
    await page.goto(`/console/#/${from}`);
    await page
      .getByTestId("page-tabs")
      .getByText(label, { exact: true })
      .click();
    await expect(page).toHaveURL(new RegExp(`#/${hash}`));
  }
});

test("playground: Esc stops a running generation", async ({ page }) => {
  const w = world();
  await install(page, w);
  await page.goto("/console/#/playground");
  const box = page.getByRole("textbox").last();
  await box.fill("hello");
  await box.press("Enter");
  await expect.poll(() => w.chatPending).toBe(true);
  await page.keyboard.press("Escape");
  await expect(page.getByText(/已停止/).first()).toBeVisible();
});

test("playground: an unloaded model offers Load in place; compare says why there is no spec switch", async ({
  page,
}) => {
  const w = world({
    models: [
      {
        id: "Qwen3.5-9B",
        type: "LLM",
        loaded: false,
        loading: false,
        pinned: false,
        size_gb: 5.8,
      },
    ],
  });
  await install(page, w);
  await page.goto("/console/#/playground");
  const load = page.getByTestId("load-inline");
  await expect(load).toContainText("載入 Qwen3.5-9B");
  await load.click();
  await expect.poll(() => w.loadCalls).toEqual(["Qwen3.5-9B"]);
  await page.getByRole("tab", { name: "比較" }).click();
  await expect(page.getByTestId("spec-note")).toContainText("沒有逐次開關");
});

test("loading model shows in the live pill, not idle", async ({ page }) => {
  const w = world({
    models: [
      {
        id: "Qwen3.5-9B",
        type: "LLM",
        loaded: false,
        loading: true,
        pinned: false,
        size_gb: 5.8,
      },
    ],
  });
  await install(page, w);
  await page.goto("/console/#/overview");
  await expect(page.getByTestId("live-phase")).toContainText("載入中");
});
