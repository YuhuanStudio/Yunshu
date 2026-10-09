import { expect, test, type Page, type Route } from "@playwright/test";

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });

const row = (o: Record<string, unknown>) => ({
  value: null,
  default: null,
  source: "default",
  stability: "stable",
  category: "server",
  description: "",
  type: "str",
  choices: [],
  applies: "live",
  minimum: null,
  secret: false,
  ...o,
});

const CONFIG = [
  row({
    name: "YUNSHU_LOG_LEVEL",
    value: "INFO",
    default: "INFO",
    type: "enum",
    choices: ["DEBUG", "INFO", "WARNING"],
    description: "Log level.",
  }),
  row({
    name: "YUNSHU_QUEUE_LIMIT",
    value: 64,
    default: 64,
    type: "int",
    minimum: 0,
    applies: "live",
    description: "Requests in flight at once.",
  }),
  row({
    name: "YUNSHU_PREFIX_MAX_ENTRIES",
    value: 64,
    default: 64,
    type: "int",
    minimum: 1,
    applies: "reload",
    description: "Prefix cache entries.",
  }),
  row({
    name: "YUNSHU_KEEP_ALIVE_TIMEOUT",
    value: 5,
    default: 5,
    type: "int",
    applies: "restart",
    source: "env",
    description: "Idle HTTP connection seconds.",
  }),
  row({
    name: "YUNSHU_AUTH_TOKEN",
    value: "***",
    default: null,
    secret: true,
    description: "Static admin token.",
  }),
  row({
    name: "YUNSHU_PREFILL_GDN",
    value: false,
    default: false,
    type: "bool",
    stability: "experimental",
    description: "Experimental switch.",
  }),
];

async function install(page: Page, opts: { admin?: boolean } = {}) {
  const admin = opts.admin !== false;
  const calls: { method: string; path: string; body: unknown }[] = [];
  await page.addInitScript(() => {
    const copied: string[] = [];
    (window as unknown as { __copied: string[] }).__copied = copied;
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: async (t: string) => void copied.push(t) },
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
  await page.route("**/openapi.json", (route) => json(route, { paths: {} }));
  await page.route("**/v1/yunshu/**", async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname.replace("/v1/yunshu", "");
    if (path === "/status") return route.fallback();
    let body: unknown = null;
    try {
      body = req.postDataJSON();
    } catch {
      /* no body */
    }
    calls.push({ method: req.method(), path, body });
    if ((!admin && req.method() !== "GET") || (!admin && path !== "/config"))
      return json(route, { detail: "no" }, 403);
    if (path === "/config" && req.method() === "GET")
      return json(route, {
        object: "yunshu.config",
        settings: CONFIG,
        warnings: [],
        experimental_count: 1,
        experimental_max: 8,
      });
    if (path === "/config" && req.method() === "PATCH") {
      const b = body as { settings: Record<string, unknown>; dry_run: boolean };
      if (
        "YUNSHU_QUEUE_LIMIT" in b.settings &&
        (b.settings.YUNSHU_QUEUE_LIMIT as number) > 1000
      )
        return json(
          route,
          {
            detail: {
              code: "invalid_settings",
              errors: { YUNSHU_QUEUE_LIMIT: "too large" },
            },
          },
          422,
        );
      const results = Object.fromEntries(
        Object.keys(b.settings).map((n) => [
          n,
          n === "YUNSHU_KEEP_ALIVE_TIMEOUT"
            ? {
                status: "overridden",
                applies: "restart",
                source: "env",
                reset: false,
              }
            : n === "YUNSHU_PREFIX_MAX_ENTRIES"
              ? {
                  status: "needs_reload",
                  applies: "reload",
                  source: "file",
                  reset: false,
                }
              : {
                  status: "applied",
                  applies: "live",
                  source: "file",
                  reset: false,
                },
        ]),
      );
      return json(route, {
        dry_run: b.dry_run,
        results,
        restart_required: false,
        reload_required: "YUNSHU_PREFIX_MAX_ENTRIES" in b.settings,
        restart: null,
      });
    }
    if (path === "/service")
      return json(route, {
        label: "ai.yunshu.server",
        plist: "/Users/me/Library/LaunchAgents/ai.yunshu.server.plist",
        installed: true,
        loaded: true,
        pid: 4242,
        state: "running",
        last_exit_code: 0,
        log: "/Users/me/.yunshu/logs/service.log",
        under_launchd: true,
        version: "0.1.4",
        uptime_s: 93784,
        cli: {
          status: "yunshu service status",
          restart: "yunshu service restart",
          install: "yunshu service install",
        },
      });
    if (path === "/service/restart")
      return json(
        route,
        {
          restarting: true,
          active_requests: 1,
          drain_timeout_s: 30,
        },
        202,
      );
    if (path === "/cors" && req.method() === "GET")
      return json(route, {
        origins: ["http://localhost:3000", "http://localhost:8000"],
        wildcard: false,
        credentials: true,
        source: "default",
        default: "http://localhost:3000,http://localhost:8000",
        warnings: [],
      });
    if (path === "/cors" && req.method() === "PATCH")
      return json(route, {
        origins: (body as { origins: string[] }).origins ?? [],
        wildcard: false,
        credentials: true,
        source: "file",
        default: "x",
        warnings: [],
      });
    if (path === "/keys")
      return json(route, {
        object: "list",
        data: [
          {
            id: "key_aaa",
            name: "筆記型電腦",
            prefix: "ysk-ab12",
            created: 1790000000,
            enabled: true,
            scopes: ["infer"],
            expires: null,
            expired: false,
            quotas: {
              requests_per_day: 1000,
              tokens_per_day: null,
              max_concurrent: 2,
            },
            last_used: Math.floor(Date.now() / 1000) - 120,
            window: { requests: 640, tokens: 120000, inflight: 0 },
          },
          {
            id: "key_bbb",
            name: "CI",
            prefix: "ysk-cd34",
            created: 1790000000,
            enabled: false,
            scopes: ["infer", "admin"],
            expires: Math.floor(Date.now() / 1000) + 86400 * 20,
            expired: false,
            quotas: {
              requests_per_day: null,
              tokens_per_day: 500000,
              max_concurrent: null,
            },
            last_used: null,
            window: { requests: 0, tokens: 0, inflight: 0 },
          },
        ],
      });
    if (path === "/usage") {
      const rows = [] as unknown[];
      for (let i = 0; i < 10; i++) {
        const day = new Date(Date.now() - i * 86400000)
          .toISOString()
          .slice(0, 10);
        rows.push({
          key: "key_aaa",
          name: "筆記型電腦",
          day,
          requests: 100 + i * 17,
          prompt_tokens: 40000 + i * 900,
          completion_tokens: 9000,
          cached_tokens: 0,
          errors: 0,
        });
      }
      return json(route, { object: "list", data: rows });
    }
    return json(route, { detail: "missing" }, 404);
  });
  return calls;
}

/** No text column may be squeezed and nothing may overflow sideways. */
async function expectNoSqueeze(page: Page, root: string) {
  const bad = await page.evaluate((sel) => {
    const scope = document.querySelector(sel);
    const out: string[] = [];
    if (!scope) return ["missing " + sel];
    for (const el of scope.querySelectorAll<HTMLElement>(
      "[data-testid=stack-row] > div:first-child",
    )) {
      if (el.getBoundingClientRect().width < 120)
        out.push(
          `narrow ${Math.round(el.getBoundingClientRect().width)}: ${el.textContent?.slice(0, 20)}`,
        );
    }
    const de = document.documentElement;
    if (de.scrollWidth > de.clientWidth + 1)
      out.push(`page overflows ${de.scrollWidth} > ${de.clientWidth}`);
    return out;
  }, root);
  expect(bad).toEqual([]);
}

test("settings rows stack on a phone: no squeezed column, no sideways overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 402, height: 874 });
  await install(page);
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  const settings = page.getByTestId("settings");
  await expect(settings.getByLabel("存取權杖")).toBeVisible();
  await expect(page.getByTestId("service-body")).toBeVisible();
  await expect(page.getByTestId("cors-editor")).toBeVisible();
  await expectNoSqueeze(page, "[data-testid=settings]");
});

test("the token is remembered only after an explicit opt-in", async ({
  page,
}) => {
  await install(page);
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  const sw = page.getByRole("switch", { name: "在此裝置記住權杖" });
  await expect(sw).toHaveAttribute("aria-checked", "false");
  expect(
    await page.evaluate(() =>
      localStorage.getItem("yunshu.console.rememberedToken"),
    ),
  ).toBeNull();
  await sw.click();
  const tokenField = page.getByLabel("存取權杖");
  await expect(async () => {
    await tokenField.fill("secret-token");
    expect(await tokenField.inputValue()).toBe("secret-token");
  }).toPass();
  await page.getByRole("button", { name: "儲存並連線" }).click();
  const stored = await page.evaluate(() =>
    localStorage.getItem("yunshu.console.rememberedToken"),
  );
  expect(stored).toContain("secret-token");
  await sw.click();
  expect(
    await page.evaluate(() =>
      localStorage.getItem("yunshu.console.rememberedToken"),
    ),
  ).toBeNull();
});

test("editing settings: inline 422 error, preview, save summary with reload and override", async ({
  page,
}) => {
  const calls = await install(page);
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  const view = page.getByTestId("config-view");
  await expect(
    view.getByRole("switch", { name: "YUNSHU_PREFILL_GDN" }),
  ).toBeVisible();
  await expect(
    view.getByText("環境變數 YUNSHU_KEEP_ALIVE_TIMEOUT 已設定"),
  ).toBeVisible();
  const queue = view.getByRole("spinbutton", { name: "YUNSHU_QUEUE_LIMIT" });
  await queue.fill("5000");
  await queue.blur();
  await view.getByRole("button", { name: "儲存", exact: true }).click();
  await expect(
    view.getByRole("alert").filter({ hasText: "too large" }),
  ).toBeVisible();
  await queue.fill("128");
  await queue.blur();
  await view.getByRole("button", { name: "預覽" }).click();
  await expect(view.getByTestId("config-summary")).toContainText("預覽");
  expect(
    calls.some(
      (c) => c.method === "PATCH" && (c.body as { dry_run: boolean }).dry_run,
    ),
  ).toBe(true);
  const prefix = view.getByRole("spinbutton", {
    name: "YUNSHU_PREFIX_MAX_ENTRIES",
  });
  await prefix.fill("32");
  await prefix.blur();
  await view.getByRole("button", { name: "儲存", exact: true }).click();
  const sum = view.getByTestId("config-summary");
  await expect(sum).toContainText("已儲存");
  await expect(sum).toContainText("重新載入模型後生效");
});

test("experimental settings need confirmation; secrets are write-only", async ({
  page,
}) => {
  const calls = await install(page);
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  const view = page.getByTestId("config-view");
  const token = view.getByLabel("YUNSHU_AUTH_TOKEN");
  await expect(token).toHaveValue("");
  await view.getByRole("switch", { name: "YUNSHU_PREFILL_GDN" }).click();
  await view.getByRole("button", { name: "儲存", exact: true }).click();
  await expect(page.getByText("修改實驗性設定")).toBeVisible();
  await page.getByRole("button", { name: "確定修改" }).click();
  await expect(view.getByTestId("config-summary")).toBeVisible();
  const patch = calls
    .filter((c) => c.method === "PATCH" && c.path === "/config")
    .at(-1);
  expect(
    (patch?.body as { confirm_experimental: boolean }).confirm_experimental,
  ).toBe(true);
});

test("service status, restart confirmation and CORS validation", async ({
  page,
}) => {
  const calls = await install(page);
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  const svc = page.getByTestId("service-section");
  await expect(svc).toContainText("4242");
  await expect(svc).toContainText("由 launchd 管理");
  await svc.getByRole("button", { name: "重新啟動" }).click();
  await page.getByRole("button", { name: "重新啟動" }).last().click();
  await expect(svc.getByText("已安排重新啟動")).toBeVisible();
  expect(calls.some((c) => c.path === "/service/restart")).toBe(true);
  const cors = page.getByTestId("cors-section");
  await cors.getByLabel("新增來源").fill("https://a.com/path");
  await cors.getByRole("button", { name: "新增" }).click();
  await expect(cors.getByRole("alert")).toContainText("不含路徑");
  await cors.getByLabel("新增來源").fill("https://app.example.com");
  await cors.getByRole("button", { name: "新增" }).click();
  await expect(cors.getByTestId("cors-origins")).toContainText(
    "https://app.example.com",
  );
  await cors.getByRole("button", { name: "儲存" }).click();
  await expect(cors.getByRole("status")).toContainText("已儲存");
});

test("old servers answer 404: calm fallback, no crash", async ({ page }) => {
  await install(page);
  await page.route("**/v1/yunshu/service", (r) =>
    json(r, { detail: "x" }, 404),
  );
  await page.route("**/v1/yunshu/cors", (r) => json(r, { detail: "x" }, 404));
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("service-section")).toContainText(
    "沒有服務管理",
  );
  await expect(page.getByTestId("cors-section")).toContainText("沒有服務管理");
});

test("api access leads with the launch one-liner and one token variable", async ({
  page,
}) => {
  await install(page);
  await page.goto("/console/#/api", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("launch-claude-code")).toContainText(
    "yunshu launch claude",
  );
  await expect(page.getByTestId("integration-codex")).toContainText(
    "YUNSHU_AUTH_TOKEN",
  );
  await expect(page.getByTestId("integrations")).not.toContainText(
    "YUNSHU_API_KEY",
  );
});

const KEYS_URL = process.env.KEYS_URL ?? "/console/#/keys";

test("keys: list with quotas, secret shown once on create, rotate, delete confirm", async ({
  page,
}) => {
  await install(page);
  let created = false;
  await page.route("**/v1/yunshu/keys", async (route) => {
    if (route.request().method() === "POST") {
      created = true;
      return json(
        route,
        {
          id: "key_new",
          name: "新金鑰",
          prefix: "ysk-zz99",
          created: 1,
          enabled: true,
          scopes: ["infer"],
          expires: null,
          expired: false,
          quotas: {
            requests_per_day: null,
            tokens_per_day: null,
            max_concurrent: null,
          },
          last_used: null,
          window: { requests: 0, tokens: 0, inflight: 0 },
          secret: "ysk-zz99-secret-value",
        },
        201,
      );
    }
    return route.fallback();
  });
  await page.goto(KEYS_URL, { waitUntil: "domcontentloaded" });
  const list = page.getByTestId("keys-list");
  await expect(list).toContainText("筆記型電腦");
  await expect(list).toContainText("640");
  await expect(list).toContainText("/ 1,000");
  await expect(page.getByTestId("keys-usage")).toBeVisible();
  await page.getByRole("button", { name: "建立金鑰" }).click();
  await page
    .getByTestId("key-form")
    .getByRole("textbox")
    .first()
    .fill("新金鑰");
  await page.getByRole("button", { name: "建立", exact: true }).click();
  const secret = page.getByTestId("key-secret");
  await expect(secret).toContainText("唯一一次");
  await expect(secret.getByRole("textbox")).toHaveValue(
    "ysk-zz99-secret-value",
  );
  expect(created).toBe(true);
  await page.getByRole("button", { name: "刪除" }).first().click();
  await expect(page.getByText("無法復原")).toBeVisible();
});

test("settings uses the standard page header; effective config shows localized descriptions, the raw name stays mono", async ({
  page,
}) => {
  await install(page);
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  await expect(
    page.getByRole("heading", { name: "設定", level: 1 }),
  ).toBeVisible();
  // One section nav: a tray on wide screens, a Select on phones; never a side column.
  const nav = page.getByTestId("settings-nav");
  await expect(nav.locator('[data-variant="tray"]')).toBeVisible();
  await expect(page.getByTestId("settings-sidebar")).toHaveCount(0);
  const view = page.getByTestId("config-view");
  const row = view.locator('tr[data-name="YUNSHU_LOG_LEVEL"]');
  await expect(row).toBeVisible();
  await expect(row.locator(".font-mono").first()).toHaveText(
    "YUNSHU_LOG_LEVEL",
  );
  // The engine's English text ("Log level.") is replaced by the zh-TW registry translation.
  await expect(row).not.toContainText("Log level.");
  await expect(row).toContainText(/[一-鿿]/);
});

test("phone settings: section picker is a Select under the standard header", async ({
  page,
}) => {
  await page.setViewportSize({ width: 402, height: 874 });
  await install(page);
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  const nav = page.getByTestId("settings-nav");
  await expect(nav.getByRole("combobox")).toBeVisible();
  await expect(nav.locator('[data-variant="tray"]')).toHaveCount(0);
});

test("phone settings: no rules between rows, no duplicated section title, the address help states the default port 8000", async ({
  page,
}) => {
  await page.setViewportSize({ width: 402, height: 874 });
  await install(page);
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  const trigger = page.getByTestId("settings-nav").getByRole("combobox");
  await expect(trigger).toBeVisible();
  // The picker names its job, the card below names the section: they never say the same words.
  await expect(trigger).not.toHaveText("引擎連線");
  const card = page.locator("#settings-connection");
  await expect(card).toContainText("服務位址");
  await expect(card).toContainText("8000");
  const rules = await card.evaluate((el) => {
    const out: string[] = [];
    for (const row of el.querySelectorAll("[data-testid=stack-row]")) {
      for (const node of [row, row.nextElementSibling]) {
        if (!node) continue;
        const cs = getComputedStyle(node);
        if (
          parseFloat(cs.borderTopWidth) > 0 ||
          parseFloat(cs.borderBottomWidth) > 0 ||
          cs.boxShadow !== "none" ||
          cs.backgroundImage !== "none"
        )
          out.push(String(node.className).slice(0, 60));
        for (const pseudo of ["::before", "::after"]) {
          const p = getComputedStyle(node, pseudo);
          if (p.content !== "none" && p.content !== "normal")
            out.push(`${pseudo} on ${String(node.className).slice(0, 40)}`);
        }
      }
    }
    return out;
  });
  expect(rules).toEqual([]);
});

test("phone keys: the usage charts fit the card without scrolling sideways", async ({
  page,
}) => {
  await page.setViewportSize({ width: 402, height: 874 });
  await install(page);
  await page.goto(KEYS_URL, { waitUntil: "domcontentloaded" });
  const usage = page.getByTestId("keys-usage");
  await expect(usage.getByRole("group").first()).toBeVisible();
  const scrolls = await usage.evaluate((el) =>
    [...el.querySelectorAll<HTMLElement>("*")]
      .filter(
        (n) =>
          /auto|scroll/.test(getComputedStyle(n).overflowX) &&
          n.scrollWidth > n.clientWidth + 1,
      )
      .map((n) => `${n.tagName}.${String(n.className).slice(0, 40)}`),
  );
  expect(scrolls).toEqual([]);
});

test("phone keys: the list is stacked cards, not a table that scrolls sideways", async ({
  page,
}) => {
  await page.setViewportSize({ width: 402, height: 874 });
  await install(page);
  await page.goto(KEYS_URL, { waitUntil: "domcontentloaded" });
  const list = page.getByTestId("keys-list");
  await expect(list).toContainText("筆記型電腦");
  await expect(page.getByTestId("keys-cards")).toBeVisible();
  const de = await page.evaluate(() => ({
    sw: document.documentElement.scrollWidth,
    cw: document.documentElement.clientWidth,
  }));
  expect(de.sw).toBeLessThanOrEqual(de.cw + 1);
  const scrolls = await list.evaluate((el) =>
    [...el.querySelectorAll<HTMLElement>("*")]
      .filter(
        (n) =>
          /auto|scroll/.test(getComputedStyle(n).overflowX) &&
          n.scrollWidth > n.clientWidth + 1,
      )
      .map((n) => n.tagName),
  );
  expect(scrolls).toEqual([]);
});
