import { expect, test, type Page } from "@playwright/test";

type Json = Record<string, unknown>;

function status(over: Json = {}): Json {
  return {
    object: "yunshu.status",
    version: "fixture-1.0",
    state: "running",
    uptime_s: 500,
    load_error: null,
    models: [
      {
        id: "Qwen3.8-27B",
        type: "VLMEngine",
        loaded: true,
        loading: false,
        pinned: false,
        size_gb: null,
      },
      {
        id: "Small",
        type: "llm",
        loaded: false,
        loading: false,
        pinned: false,
      },
    ],
    memory: { active_gb: 12, cache_gb: 2, peak_gb: 14, total_gb: 64 },
    requests: { active: 0, queued: 0, prefill: 0, decode: 0, items: [] },
    last: {
      request_id: "req_last",
      prompt_tokens: 100,
      completion_tokens: 20,
      cached_tokens: 50,
      prefill_tps: 700,
      decode_tps: 25,
      ttft_ms: 131,
      t: 1_700_000_000,
    },
    throughput: {
      window_s: 60,
      requests: 3,
      prompt_tokens: 1,
      completion_tokens: 1,
      live_decode_tps: null,
      mean_prefill_tps: 800,
      mean_decode_tps: 48,
    },
    ...over,
  };
}

const prefilling = status({
  requests: {
    active: 1,
    queued: 0,
    prefill: 1,
    decode: 0,
    items: [
      {
        request_id: "qa-prefill-01",
        elapsed_s: 2,
        phase: "prefill",
        prompt_tokens: 1000,
        processed_tokens: 340,
        tokens_per_second: 170,
      },
    ],
  },
});

async function install(page: Page, body: () => Json, history?: Json | null) {
  const asked: string[] = [];
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    asked.push(route.request().method() + " " + path);
    if (path === "/v1/yunshu/status") return route.fulfill({ json: body() });
    if (path === "/v1/yunshu/history")
      return history
        ? route.fulfill({ json: history })
        : route.fulfill({ status: 404, json: { detail: "no" } });
    return route.fulfill({ status: 404, json: { detail: "fixture" } });
  });
  return asked;
}

test.describe("overview tells the truth about the engine", () => {
  test("window label follows throughput.window_s, never five minutes", async ({
    page,
  }) => {
    await install(page, () =>
      status({
        throughput: { ...(status().throughput as Json), window_s: 30 },
      }),
    );
    await page.goto("/console/#/overview");
    const overview = page.getByTestId("overview");
    await expect(overview.getByTestId("speed-pair")).toContainText(
      "近 30 秒均值",
    );
    await expect(overview).not.toContainText("近 5 分鐘平均");
    await expect(
      overview.getByText("近 30 秒結束", { exact: false }),
    ).toBeVisible();
  });

  test("a request in prefill is shown as prefill, never as idle", async ({
    page,
  }) => {
    await install(page, () => prefilling);
    await page.goto("/console/#/overview");
    const decode = page.getByTestId("speed-decode");
    await expect(decode).toContainText("預填中");
    await expect(decode.getByTestId("speed-decode-label")).toHaveAttribute(
      "title",
      "預填中，尚無解碼速度",
    );
    await expect(decode).not.toContainText("閒置");
    const strip = page.getByTestId("state-strip");
    await expect(strip.locator("[data-phase=prefill]")).toHaveAttribute(
      "data-lit",
      "true",
    );
    // Idle is no pill of its own: it is the absence of a lit phase.
    await expect(strip.locator("[data-phase=idle]")).toHaveCount(0);
    await expect(strip.getByTestId("state-strip-detail")).toContainText("34%");
    // The percentage is in the sentence; no bar is glued to the card edge (it read as a stray line).
    await expect(strip.getByRole("progressbar")).toHaveCount(0);
    await expect(page.getByTestId("live-phase")).toContainText("預填");
    await expect(page.getByTestId("live-phase")).toContainText("34%");
  });

  test("idle: top bar says 閒置 without a stale tok/s, the card labels the last request", async ({
    page,
  }) => {
    await install(page, () => status());
    await page.goto("/console/#/overview");
    const pill = page.getByTestId("live-phase");
    await expect(pill).toContainText("閒置");
    await expect(pill).not.toContainText("tok/s");
    await expect(page.getByTestId("speed-decode-label")).toContainText(
      "最近一筆",
    );
    await expect(page.getByTestId("speed-decode")).toContainText("25.0");
    await expect.poll(() => page.title()).not.toContain("tok/s");
    // The pill is on every page.
    await page.goto("/console/#/requests");
    await expect(page.getByTestId("live-phase")).toContainText("閒置");
  });

  test("live decode is a labelled aggregate and the pill shows the number", async ({
    page,
  }) => {
    await install(page, () =>
      status({
        requests: { active: 2, queued: 0, prefill: 0, decode: 2, items: [] },
        throughput: { ...(status().throughput as Json), live_decode_tps: 80 },
      }),
    );
    await page.goto("/console/#/overview");
    await expect(page.getByTestId("speed-decode-label")).toContainText(
      "即時合計",
    );
    await expect(page.getByTestId("live-phase")).toContainText("80.0 tok/s");
  });

  test("null size_gb or a malformed optional field does not turn the engine offline", async ({
    page,
  }) => {
    await install(page, () =>
      status({
        models: [
          {
            id: "A",
            type: "llm",
            loaded: true,
            loading: false,
            pinned: false,
            size_gb: null,
            idle_s: "x",
          },
        ],
        memory: { active_gb: null, total_gb: "n/a", cache_gb: 2 },
      }),
    );
    await page.goto("/console/#/overview");
    await expect(page.getByTestId("live-phase")).toContainText("閒置");
    await expect(page.getByText("無法連接引擎")).toHaveCount(0);
  });

  test("percentiles stay hidden below 20 samples", async ({ page }) => {
    await install(page, () => status());
    await page.goto("/console/#/overview");
    const latency = page.getByTestId("latency-panel");
    await expect(latency.getByTestId("latency-last")).toContainText("131");
    await expect(latency.getByTestId("latency-p50")).toContainText("—");
    await expect(latency.getByTestId("latency-p95")).toContainText("—");
    await expect(latency).toContainText("滿 20 筆才顯示 P50 / P95");
  });
});

test.describe("engine-side history", () => {
  const now = () => Date.now() / 1000;
  const rows = (n: number) => {
    const t0 = now() - n * 5;
    const col = (f: (i: number) => number | null) =>
      Array.from({ length: n }, (_, i) => f(i));
    return {
      object: "yunshu.history",
      enabled: true,
      interval_s: 5,
      series: {
        t: col((i) => t0 + i * 5),
        decode_tps: col((i) => (i % 4 ? 30 + i : null)),
        prefill_tps: col(() => null),
        requests_active: col(() => 1),
        queued: col(() => 0),
        active_gb: col(() => 10),
        cache_gb: col(() => 1),
      },
    };
  };

  test("charts start filled from the engine history and say so", async ({
    page,
  }) => {
    await install(page, () => status(), rows(60));
    await page.goto("/console/#/overview");
    const overview = page.getByTestId("overview");
    await expect(overview).toContainText("含引擎端歷史");
    await expect(overview).toContainText(/ 6\d 筆 /);
  });

  test("an older server without the route falls back to live polls", async ({
    page,
  }) => {
    await install(page, () => status(), null);
    await page.goto("/console/#/overview");
    await expect(page.getByTestId("overview")).toContainText(
      "此引擎沒有提供歷史",
    );
    await expect(page.getByTestId("live-phase")).toContainText("閒置");
  });
});

test.describe("shell accessibility", () => {
  test("skip link is the first tab stop and moves focus to main", async ({
    page,
    browserName,
  }) => {
    // Safari does not Tab to buttons or focus them on click by default.
    test.skip(browserName === "webkit");
    await install(page, () => status());
    await page.goto("/console/#/overview");
    await expect(page.getByTestId("overview")).toBeVisible();
    await page.keyboard.press("Tab");
    const skip = page.getByRole("link", { name: "跳到主要內容" });
    await expect(skip).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page.locator("#main-content")).toBeFocused();
  });

  test("sidebar comes before the page in tab order and landmarks are valid", async ({
    page,
  }) => {
    await install(page, () => status());
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/console/#/overview");
    await expect(page.getByTestId("overview")).toBeVisible();
    const order = await page.evaluate(() => {
      const nav = document.querySelector("nav[aria-label='控制台導覽']");
      const main = document.querySelector("main");
      return {
        navFirst:
          !!nav &&
          !!main &&
          !!(
            nav.compareDocumentPosition(main) & Node.DOCUMENT_POSITION_FOLLOWING
          ),
        navs: document.querySelectorAll("nav[aria-label='控制台導覽']").length,
        nestedNav: document.querySelectorAll("nav nav").length,
        mains: document.querySelectorAll("main").length,
        asideNav: document.querySelectorAll("aside[role=navigation]").length,
        banner: document.querySelectorAll("body > div header, header").length,
        headerInMain: document.querySelectorAll("main header").length,
      };
    });
    expect(order).toMatchObject({
      navFirst: true,
      navs: 1,
      nestedNav: 0,
      mains: 1,
      asideNav: 0,
      headerInMain: 0,
    });
    expect(order.banner).toBeGreaterThan(0);
  });

  test("sidebar footer buttons are named by their visible text; no clickToCopy key leaks", async ({
    page,
  }) => {
    await install(page, () => status());
    await page.goto("/console/#/overview");
    const settings = page.getByRole("button", { name: /^開啟設定/ });
    await expect(settings).toContainText("fixture");
    await expect(settings).not.toHaveAttribute("aria-label", /.+/);
    await page.goto("/console/#/models/Qwen3.8-27B");
    await expect(
      page.locator("[title='clickToCopy'], [aria-label='clickToCopy']"),
    ).toHaveCount(0);
  });

  test("theme toggle shows a keyboard focus ring", async ({
    page,
    browserName,
  }) => {
    // Safari does not Tab to buttons or focus them on click by default.
    test.skip(browserName === "webkit");
    await install(page, () => status());
    await page.goto("/console/#/overview");
    const toggle = page.getByRole("button", { name: /切換(深|淺)色/ });
    await toggle.focus();
    await page.keyboard.press("Shift+Tab");
    await page.keyboard.press("Tab");
    await expect(toggle).toBeFocused();
    const outline = await toggle.evaluate((el) => {
      const cs = getComputedStyle(el);
      return { style: cs.outlineStyle, width: cs.outlineWidth };
    });
    expect(outline.style).not.toBe("none");
    expect(parseFloat(outline.width)).toBeGreaterThanOrEqual(2);
  });

  test("the polite live region does not chatter every poll", async ({
    page,
  }) => {
    await install(page, () => status());
    await page.goto("/console/#/overview");
    await expect(page.getByTestId("overview")).toBeVisible();
    const live = await page.evaluate(() => ({
      polite: [...document.querySelectorAll("[aria-live=polite]")].filter(
        (e) => (e.textContent ?? "").trim() !== "",
      ).length,
      statusRegions: [...document.querySelectorAll("[role=status]")].map((e) =>
        (e.textContent ?? "").slice(0, 20),
      ),
    }));
    expect(live.polite).toBe(0);
    expect(live.statusRegions.filter((t) => t.includes("收集採樣中"))).toEqual(
      [],
    );
  });

  test("Escape in the command palette returns focus to the opener", async ({
    page,
    browserName,
  }) => {
    // Safari does not Tab to buttons or focus them on click by default.
    test.skip(browserName === "webkit");
    await install(page, () => status());
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/console/#/overview");
    const opener = page.getByRole("button", { name: /搜尋/ });
    await opener.focus();
    await opener.click();
    await expect(page.getByRole("combobox")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("combobox")).toHaveCount(0);
    await expect(opener).toBeFocused();
  });
});

test.describe("route splitting", () => {
  test("the overview loads no syntax-highlighter, diagram or math chunks", async ({
    page,
  }) => {
    await install(page, () => status());
    const heavy: string[] = [];
    page.on("request", (r) => {
      const u = new URL(r.url()).pathname;
      if (
        /(mermaid|cytoscape|katex|shiki|\/cpp-|\/content[-.])/i.test(u) &&
        u.endsWith(".js")
      )
        heavy.push(u);
    });
    await page.goto("/console/#/overview");
    await expect(page.getByTestId("overview")).toBeVisible();
    await page.waitForTimeout(800);
    expect(heavy).toEqual([]);
  });
});
