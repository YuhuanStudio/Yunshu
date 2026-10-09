import { expect, test, type Page, type Route } from "@playwright/test";
import { mkdirSync } from "node:fs";

const SHOTS = process.env.R6_SHOTS;

const model = (id: string, over: Record<string, unknown> = {}) => ({
  id,
  type: "text",
  loaded: false,
  loading: false,
  pinned: false,
  expires_in_s: null,
  ...over,
});
const GB = 2 ** 30; // binary, labelled GB as macOS does

type World = {
  models: Record<string, unknown>[];
  downloads: Record<string, unknown>[];
  local: Record<string, unknown>[];
  fit: Record<string, Record<string, unknown>>;
  cache: Record<string, unknown> | null;
  postDownload: (body: Record<string, unknown>) => {
    code: number;
    body: unknown;
  };
  calls: string[];
  support: { downloads: boolean; local: boolean; fit: boolean; cache: boolean };
};

const job = (over: Record<string, unknown>) => ({
  id: "j1",
  repo: "mlx-community/Qwen3-4B-4bit",
  revision: null,
  allow_patterns: null,
  state: "running",
  error: null,
  path: "/models/mlx-community/Qwen3-4B-4bit",
  bytes_total: 8.1 * GB,
  bytes_done: 3.2 * GB,
  files_total: 12,
  files_done: 3,
  active_files: ["model-00004-of-00005.safetensors"],
  rate_bps: 45 * 2 ** 20,
  eta_s: 108,
  created: 1,
  started: 1,
  finished: null,
  registered: false,
  already_present: false,
  ...over,
});

function world(): World {
  return {
    models: [
      model("org/big", { loaded: true, pinned: true, size_gb: 25 }),
      model("org/tight", { size_gb: 20 }),
      model("org/huge", { size_gb: 90 }),
      model("org/fits", { size_gb: 4 }),
      model("org/booting", { loading: true, size_gb: 8 }),
    ],
    downloads: [
      job({}),
      job({
        id: "j2",
        repo: "org/broken",
        state: "failed",
        error: "ConnectionError: reset",
        bytes_done: 1 * GB,
        rate_bps: null,
        eta_s: null,
      }),
      job({
        id: "j3",
        repo: "org/paused",
        state: "cancelled",
        bytes_done: 2 * GB,
        rate_bps: null,
        eta_s: null,
      }),
      job({
        id: "j4",
        repo: "org/finished",
        state: "done",
        bytes_done: 2 * GB,
        bytes_total: 2 * GB,
        files_done: 4,
        files_total: 4,
        registered: true,
        rate_bps: null,
        eta_s: null,
      }),
    ],
    local: [
      {
        id: "org/on-disk",
        path: "/models/org/on-disk",
        source: "models_dir",
        size_bytes: 4.4 * GB,
        model_type: "qwen3",
        kind: "llm",
        architecture: "Qwen3",
        parameters: "8B",
        quantization: { bits: 4, group_size: 64 },
        context_length: 32768,
        capabilities: ["tools", "reasoning"],
        complete: true,
        complete_reason: null,
        registered_as: null,
        loaded: false,
      },
      {
        id: "org/half",
        path: "/models/org/half",
        source: "models_dir",
        size_bytes: 1.1 * GB,
        model_type: "qwen3",
        capabilities: [],
        complete: false,
        complete_reason: "missing model-00002-of-00002.safetensors",
        registered_as: null,
        loaded: false,
      },
      {
        id: "org/big",
        path: "/models/org/big",
        source: "models_dir",
        size_bytes: 25 * GB,
        complete: true,
        registered_as: "org/big",
        loaded: true,
      },
    ],
    fit: {
      "org/tight": {
        model: "org/tight",
        verdict: "tight",
        reason: "fits only after evicting org/big",
        weights_bytes: 20 * GB,
        kv_reserve_bytes: 2 * GB,
        needed_bytes: 22 * GB,
        budget_bytes: 48 * GB,
        used_bytes: 36 * GB,
        free_bytes: 12 * GB,
        free_bytes_after_evict: 37 * GB,
        would_evict: ["org/other-idle"],
        loaded: false,
        basis: { estimated: true },
      },
      "org/huge": {
        model: "org/huge",
        verdict: "wont_fit",
        reason: "needs 99 GB",
        weights_bytes: 90 * GB,
        kv_reserve_bytes: 9 * GB,
        needed_bytes: 99 * GB,
        budget_bytes: 48 * GB,
        used_bytes: 25 * GB,
        free_bytes: 23 * GB,
        would_evict: [],
        loaded: false,
        basis: { estimated: true },
      },
      "org/fits": {
        model: "org/fits",
        verdict: "fits",
        weights_bytes: 4 * GB,
        kv_reserve_bytes: 0.4 * GB,
        needed_bytes: 4.4 * GB,
        budget_bytes: 48 * GB,
        used_bytes: 25 * GB,
        free_bytes: 23 * GB,
        would_evict: [],
        loaded: false,
        basis: { estimated: true },
      },
    },
    cache: {
      enabled: true,
      caches: [
        {
          model: "org/big",
          tiers: [
            {
              name: "ram",
              used_bytes: 2.4 * GB,
              cap_bytes: 8 * GB,
              entries: 14,
              hits: 80,
            },
            {
              name: "warm",
              used_bytes: 0.6 * GB,
              cap_bytes: 4 * GB,
              entries: 5,
              hits: 6,
              mode: "int8",
            },
            {
              name: "ssd",
              used_bytes: 12 * GB,
              cap_bytes: 64 * GB,
              entries: 31,
              hits: 9,
            },
          ],
          lookups: { hit: 95, miss: 25, by_tier: { ram: 80, warm: 6, ssd: 9 } },
          entries: Array.from({ length: 22 }, (_, i) => ({
            key: (0xa1b2c3d4 + i * 4097).toString(16).padStart(8, "0"),
            tokens: 1000 + i * 700,
            bytes: (200 + i * 30) * 1e6,
            tier: i % 4 === 0 ? "warm" : "ram",
            lru_rank: i,
            hits: i % 5,
            last_hit_age_s: i % 3 === 0 ? null : i * 41,
          })),
          entries_truncated: false,
        },
      ],
    },
    postDownload: () => ({
      code: 202,
      body: job({
        id: "jn",
        repo: "org/new",
        state: "queued",
        bytes_done: 0,
        rate_bps: null,
        eta_s: null,
      }),
    }),
    calls: [],
    support: { downloads: true, local: true, fit: true, cache: true },
  };
}

const status = (models: unknown[]) => ({
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "ready",
  uptime_s: 100,
  load_error: null,
  models,
  memory: {
    active_gb: 25,
    cache_gb: 1,
    peak_gb: 30,
    total_gb: 48,
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

const ledger = {
  object: "yunshu.memory",
  // A newer engine: binary GB plus exact bytes (bytes win over the GB fields).
  total_gb: 48,
  total_bytes: 48 * GB,
  free_gb: 23,
  host: { pressure_level: "normal", swap_used_gb: 0 },
  mlx: {
    active_gb: 25,
    cache_gb: 1,
    peak_gb: 30,
    recommended_working_set_gb: 40,
  },
  owners: [
    {
      kind: "weights",
      id: "org/big",
      bytes: 22 * GB,
      gb: 22,
      reclaimable: false,
      estimated: false,
      source: "p",
    },
    {
      kind: "apc_ram",
      id: "org/big",
      bytes: 2.4 * GB,
      gb: 2.4,
      reclaimable: true,
      estimated: false,
      source: "p",
    },
    {
      kind: "apc_warm",
      id: "org/big",
      bytes: 0.6 * GB,
      gb: 0.6,
      reclaimable: true,
      estimated: false,
      source: "p",
    },
  ],
  attribution_overshoot_gb: null,
  limits: { apc_max_gb: null, apc_warm_max_gb: null, guard_margin_pct: null },
};

async function install(page: Page, w: World) {
  await page.route("**/v1/**", async (route: Route) => {
    const req = route.request(),
      url = new URL(req.url()),
      path = url.pathname,
      method = req.method();
    const json = (code: number, body: unknown) =>
      route.fulfill({
        status: code,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    const nf = () => json(404, { detail: "Not Found" });
    w.calls.push(`${method} ${path}`);
    if (path === "/v1/yunshu/status") return json(200, status(w.models));
    if (path === "/v1/yunshu/memory") return json(200, ledger);
    if (path === "/v1/yunshu/downloads") {
      if (!w.support.downloads) return nf();
      if (method === "POST") {
        const r = w.postDownload(JSON.parse(req.postData() ?? "{}"));
        return json(r.code, r.body);
      }
      return json(200, {
        downloads: w.downloads,
        active: 1,
        free_bytes: 212 * GB,
        models_dir: "/Users/me/.yunshu/models",
      });
    }
    if (path.startsWith("/v1/yunshu/downloads/") && method === "DELETE") {
      if (!w.support.downloads) return nf();
      const id = path.split("/").pop();
      const j = w.downloads.find((d) => d.id === id)!;
      j.state = "cancelled";
      return json(200, j);
    }
    if (path === "/v1/yunshu/models/local")
      return w.support.local
        ? json(200, {
            models: w.local,
            total_bytes: 30 * GB,
            models_dir: "/models",
            free_bytes: 212 * GB,
          })
        : nf();
    if (path.startsWith("/v1/yunshu/models/") && path.endsWith("/fit")) {
      if (!w.support.fit) return nf();
      const id = decodeURIComponent(
        path.slice("/v1/yunshu/models/".length, -4),
      );
      return w.fit[id] ? json(200, w.fit[id]) : nf();
    }
    if (path === "/v1/yunshu/cache/tiers" && method === "GET")
      return w.support.cache ? json(200, w.cache) : nf();
    if (path === "/v1/yunshu/cache/tiers/clear") {
      if (!w.support.cache) return nf();
      const body = JSON.parse(req.postData() ?? "{}");
      return json(200, {
        cleared: [
          {
            model: "org/big",
            tier: body.tier,
            entries: 14,
            freed_bytes: 2.4 * GB,
          },
        ],
        freed_bytes: 2.4 * GB,
      });
    }
    if (path === "/v1/models/load") return json(200, { status: "loaded" });
    if (path.startsWith("/v1/models/")) return json(200, { id: "card" });
    return nf();
  });
}

async function openPage(page: Page, name: "downloads" | "cache") {
  await page.goto(`/console/#/${name}`, { waitUntil: "domcontentloaded" });
  await page.getByTestId(name).waitFor();
}
const openModels = async (page: Page, sub = "") => {
  await page.goto(`/console/#/models${sub}`, { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("models")).toBeVisible();
};
const noOverflow = async (page: Page) =>
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth - innerWidth,
    ),
  ).toBeLessThanOrEqual(0);

test.describe("downloads", () => {
  test("progress, rate, ETA, files, cancel and resume", async ({ page }) => {
    const w = world();
    await install(page, w);
    await openPage(page, "downloads");
    const rows = page.getByTestId("download-row");
    await expect(rows).toHaveCount(4);
    const running = rows.filter({ hasText: "Qwen3-4B-4bit" });
    await expect(running.getByTestId("download-stats")).toContainText("3.2GB");
    await expect(running.getByTestId("download-stats")).toContainText(
      "45.0MB/s",
    );
    await expect(running.getByTestId("download-stats")).toContainText("3 / 12");
    await expect(running.getByRole("progressbar")).toBeVisible();
    // finished + registered links to the model
    await expect(
      rows.filter({ hasText: "org/finished" }).getByRole("link"),
    ).toHaveAttribute("href", /#\/models\/org%2Ffinished/);
    // resume re-posts the same repo
    let posted: Record<string, unknown> | null = null;
    w.postDownload = (b) => (
      (posted = b),
      { code: 202, body: job({ id: "jr", repo: String(b.repo) }) }
    );
    await rows
      .filter({ hasText: "org/paused" })
      .getByRole("button", { name: "繼續下載" })
      .click();
    await expect.poll(() => posted?.repo).toBe("org/paused");
    // cancel the running one
    await running.getByRole("button", { name: "取消" }).click();
    await expect
      .poll(() => w.calls.includes("DELETE /v1/yunshu/downloads/j1"))
      .toBe(true);
    await noOverflow(page);
  });

  test("the 507 disk shortfall is explained, with both numbers", async ({
    page,
  }) => {
    const w = world();
    w.postDownload = () => ({
      code: 507,
      body: {
        detail: {
          message: "x",
          needed_bytes: 40 * GB,
          free_bytes: 6 * GB,
          path: "/m",
        },
      },
    });
    await install(page, w);
    await openPage(page, "downloads");
    await page.getByLabel("儲存庫").fill("org/too-big");
    await page.getByRole("button", { name: "開始下載" }).click();
    const banner = page.getByText("磁碟空間不足");
    await expect(banner).toBeVisible();
    await expect(page.getByText(/需要 40.0GB，目前只剩 6.0GB/)).toBeVisible();
    await expect(page.getByLabel("儲存庫")).toHaveValue("org/too-big"); // kept so it can be fixed
  });

  test("a model that is already on disk says so", async ({ page }) => {
    const w = world();
    w.postDownload = () => ({
      code: 202,
      body: job({
        id: "jp",
        repo: "org/have",
        state: "done",
        already_present: true,
        registered: false,
      }),
    });
    await install(page, w);
    await openPage(page, "downloads");
    await page.getByLabel("儲存庫").fill("org/have");
    await page.getByRole("button", { name: "開始下載" }).click();
    await expect(
      page.getByText("org/have 已在磁碟上，不需要再下載。"),
    ).toBeVisible();
  });

  test("an invalid repo keeps the button disabled with a reason", async ({
    page,
  }) => {
    await install(page, world());
    await openPage(page, "downloads");
    await page.getByLabel("儲存庫").fill("nope");
    await expect(page.getByRole("button", { name: "開始下載" })).toBeDisabled();
    await expect(page.getByText("格式應為 org/name。")).toBeVisible();
  });
});

test.describe("models library", () => {
  test("on-disk models that are not registered appear with facts and status", async ({
    page,
  }) => {
    await install(page, world());
    await openModels(page);
    const rows = page.getByTestId("local-row");
    await expect(rows).toHaveCount(2); // the registered one is not repeated
    await expect(rows.filter({ hasText: "org/on-disk" })).toContainText(
      "4 位元量化",
    );
    await expect(rows.filter({ hasText: "org/on-disk" })).toContainText(
      "上下文 32,768",
    );
    await expect(rows.filter({ hasText: "org/on-disk" })).toContainText(
      "檔案完整",
    );
    const half = rows.filter({ hasText: "org/half" });
    await expect(half).toContainText("檔案不完整");
    await expect(half).toContainText("missing model-00002");
    await expect(
      half.getByRole("button", { name: "註冊並載入" }),
    ).toBeDisabled();
    await half.getByTestId("reasoned").focus();
    await expect(page.getByRole("tooltip").first()).toContainText(
      "檔案不完整，請先重新下載",
    );
  });

  test("Load asks first when the fit is tight, and shows what would be evicted", async ({
    page,
  }) => {
    const w = world();
    await install(page, w);
    await openModels(page);
    const row = page.getByRole("row").filter({ hasText: "tight" });
    await row.getByRole("button", { name: "載入", exact: true }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog.getByTestId("fit-panel")).toHaveAttribute(
      "data-verdict",
      "tight",
    );
    await expect(dialog.getByTestId("fit-evict")).toContainText(
      "org/other-idle",
    );
    expect(w.calls).not.toContain("POST /v1/models/load");
    await dialog.getByRole("button", { name: "仍然載入" }).click();
    await expect
      .poll(() => w.calls.includes("POST /v1/models/load"))
      .toBe(true);
  });

  test("a model that will not fit cannot be loaded from the dialog", async ({
    page,
  }) => {
    const w = world();
    await install(page, w);
    await openModels(page);
    await page
      .getByRole("row")
      .filter({ hasText: "huge" })
      .getByRole("button", { name: "載入", exact: true })
      .click();
    const dialog = page.getByRole("dialog");
    await expect(dialog.getByTestId("fit-panel")).toHaveAttribute(
      "data-verdict",
      "wont_fit",
    );
    await expect(
      dialog.getByRole("button", { name: "仍然載入" }),
    ).toBeDisabled();
    expect(w.calls).not.toContain("POST /v1/models/load");
  });

  test("a model that fits loads straight away", async ({ page }) => {
    const w = world();
    await install(page, w);
    await openModels(page);
    await page
      .getByRole("row")
      .filter({ hasText: "fits" })
      .getByRole("button", { name: "載入", exact: true })
      .click();
    await expect
      .poll(() => w.calls.includes("POST /v1/models/load"))
      .toBe(true);
    await expect(page.getByRole("dialog")).toHaveCount(0);
  });

  test("detail page shows the fit before loading", async ({ page }) => {
    await install(page, world());
    await openModels(page, "/org%2Ftight");
    const card = page.getByTestId("fit-card");
    await expect(card.getByTestId("fit-panel")).toHaveAttribute(
      "data-verdict",
      "tight",
    );
    await expect(card).toContainText("22.0GB");
  });

  test("loading shows elapsed time and never reads as loaded; pinned unload says why", async ({
    page,
  }) => {
    await install(page, world());
    await openModels(page);
    const booting = page.getByRole("row").filter({ hasText: "booting" });
    await expect(booting).toContainText("載入中");
    await expect(booting).toContainText(/已耗時/);
    await expect(booting).not.toContainText("已載入");
    const big = page.getByRole("row").filter({ hasText: "big" });
    await expect(big.getByRole("button", { name: "卸載" })).toBeDisabled();
    await big.getByTestId("reasoned").focus();
    await expect(page.getByRole("tooltip").first()).toContainText("已固定保留");
  });

  test("download entry point and running-download notice", async ({ page }) => {
    await install(page, world());
    await openModels(page);
    await expect(page.getByTestId("downloads-running")).toContainText(
      "1 個模型下載中",
    );
    await expect(page.getByRole("button", { name: "下載模型" })).toBeVisible();
  });

  test("an older server (404 everywhere) hides the new features calmly", async ({
    page,
  }) => {
    const w = world();
    w.support = { downloads: false, local: false, fit: false, cache: false };
    await install(page, w);
    await openModels(page);
    await expect(page.getByRole("button", { name: "下載模型" })).toHaveCount(0);
    await expect(page.getByTestId("local-inventory")).toHaveCount(0);
    await expect(page.getByTestId("downloads-running")).toHaveCount(0);
    // load still works, with no dialog
    await page
      .getByRole("row")
      .filter({ hasText: "fits" })
      .getByRole("button", { name: "載入", exact: true })
      .click();
    await expect
      .poll(() => w.calls.includes("POST /v1/models/load"))
      .toBe(true);
    await openPage(page, "downloads");
    await expect(page.getByTestId("downloads-unsupported")).toBeVisible();
    await openPage(page, "cache");
    await expect(page.getByTestId("cache-unsupported")).toBeVisible();
  });
});

test.describe("cache", () => {
  test("tiers, lookups, entries and a confirmed clear that reports freed bytes", async ({
    page,
  }) => {
    const w = world();
    await install(page, w);
    await openPage(page, "cache");
    const tiers = page.getByTestId("cache-tier");
    await expect(tiers).toHaveCount(3);
    await expect(tiers.filter({ hasText: "記憶體層" })).toContainText("2.4GB");
    await expect(tiers.filter({ hasText: "SSD 層" })).toContainText("31");
    await expect(page.getByTestId("cache-lookups")).toContainText("95");
    await expect(page.getByTestId("cache-lookups")).toContainText("79.2%");
    // top entries only, never text
    await expect(page.getByTestId("cache-entry")).toHaveCount(15);
    await expect(
      page.getByText("只顯示雜湊識別碼，不含對話內容。"),
    ).toBeVisible();
    // share of memory from the ledger: (2.4 + 0.6) / 48
    await expect(page.getByText("6.3%")).toBeVisible();
    // clear needs a confirmation
    await tiers
      .filter({ hasText: "記憶體層" })
      .getByRole("button", { name: "清除" })
      .click();
    expect(w.calls).not.toContain("POST /v1/yunshu/cache/tiers/clear");
    await page
      .getByRole("dialog")
      .getByRole("button", { name: "清除", exact: true })
      .click();
    await expect(page.getByTestId("cache-cleared")).toContainText("釋放 2.4GB");
    await noOverflow(page);
  });

  test("the memory ledger links to the cache", async ({ page }) => {
    await install(page, world());
    await openModels(page, "/org%2Fbig");
    await expect(page.getByTestId("ledger-cache-link").first()).toHaveAttribute(
      "href",
      "#/cache",
    );
  });
});

test.describe("screenshots", () => {
  test.skip(!SHOTS, "set R6_SHOTS to a directory to capture");
  for (const [w, h] of [
    [1440, 1000],
    [390, 900],
  ] as const)
    for (const dark of [false, true])
      test(`pages at ${w} ${dark ? "dark" : "light"}`, async ({
        page,
      }, info) => {
        mkdirSync(SHOTS!, { recursive: true });
        await page.setViewportSize({ width: w, height: h });
        await page.addInitScript((d) => {
          try {
            localStorage.setItem("yunshu.console.theme", d ? "dark" : "light");
          } catch {}
          document.documentElement.classList.toggle("dark", d);
        }, dark);
        await install(page, world());
        const tag = `${info.project.name}-${w}-${dark ? "dark" : "light"}`;
        await openPage(page, "downloads");
        await page.getByRole("progressbar").first().waitFor();
        await page.screenshot({
          path: `${SHOTS}/downloads-${tag}.png`,
          fullPage: true,
        });
        await openPage(page, "cache");
        await page.getByTestId("cache-entry").first().waitFor();
        await page.screenshot({
          path: `${SHOTS}/cache-${tag}.png`,
          fullPage: true,
        });
        await openModels(page);
        await page.getByTestId("local-row").first().waitFor();
        await page.screenshot({
          path: `${SHOTS}/models-${tag}.png`,
          fullPage: true,
        });
      });
});

test.describe("iPhone width (402x874)", () => {
  test.use({ viewport: { width: 402, height: 874 } });
  const inside = async (page: Page, loc: ReturnType<Page["locator"]>) => {
    const b = await loc.boundingBox();
    expect(b).not.toBeNull();
    expect(b!.x).toBeGreaterThanOrEqual(0);
    expect(b!.x + b!.width).toBeLessThanOrEqual(402);
  };
  test("models memory summary stacks left-aligned", async ({ page }) => {
    await install(page, world());
    await openModels(page);
    const card = page.getByTestId("memory-summary");
    const kids = card.locator(":scope > *");
    await expect(kids).toHaveCount(3);
    const xs = await kids.evaluateAll((els) =>
      els.map((e) => Math.round(e.getBoundingClientRect().left)),
    );
    expect(new Set(xs).size).toBe(1); // one left edge
    const inner = await card
      .locator("p")
      .evaluateAll((els) =>
        els.map((e) => Math.round(e.getBoundingClientRect().left)),
      );
    expect(Math.max(...inner) - Math.min(...inner)).toBeLessThanOrEqual(1);
    await noOverflow(page);
    for (const name of ["下載模型"])
      await inside(page, page.getByRole("button", { name }));
    await inside(
      page,
      page
        .getByTestId("local-row")
        .first()
        .getByRole("button", { name: "註冊並載入" }),
    );
  });
  test("downloads and cache fit and every action is reachable", async ({
    page,
  }) => {
    await install(page, world());
    await openPage(page, "downloads");
    await noOverflow(page);
    for (const name of ["開始下載", "取消", "繼續下載"])
      await inside(page, page.getByRole("button", { name }).first());
    await openPage(page, "cache");
    await page.getByTestId("cache-tier").first().waitFor();
    await noOverflow(page);
    await page
      .getByTestId("cache-tier")
      .first()
      .getByRole("button", { name: "清除" })
      .scrollIntoViewIfNeeded();
  });
});
