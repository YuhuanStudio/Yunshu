import { expect, test, type Page } from "@playwright/test";

// The docs section: pages render from the compiled MDX, links and anchors work, the command
// palette searches them, and the console pages link to the page that explains them.

const status = {
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "ready",
  uptime_s: 100,
  load_error: null,
  models: [],
  memory: { active_gb: 1, cache_gb: 0, peak_gb: 1, total_gb: 128 },
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

async function open(page: Page, hash: string, locale?: string) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.addInitScript((l) => {
    localStorage.setItem("yunshu.console.url", location.origin);
    if (l) localStorage.setItem("yunshu.console.locale", l);
  }, locale ?? null);
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") return route.fulfill({ json: status });
    return route.fulfill({ status: 404, json: { error: { message: "n/a" } } });
  });
  await page.goto(`/console/#/${hash}`, { waitUntil: "domcontentloaded" });
  return errors;
}

const article = (page: Page) => page.getByTestId("docs-article");

test("the docs home renders in the console language with the page tree", async ({
  page,
}) => {
  const errors = await open(page, "docs");
  await expect(article(page).getByRole("heading", { level: 1 })).toHaveText(
    "簡介",
  );
  const nav = page.getByTestId("docs-nav");
  await expect(
    nav.getByRole("link", { name: "Chat completions" }),
  ).toBeVisible();
  await expect(nav.getByRole("link", { name: "簡介" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  expect(errors).toEqual([]);
});

test("the sidebar and in-page links navigate between pages", async ({
  page,
}) => {
  await open(page, "docs");
  await page
    .getByTestId("docs-nav")
    .getByRole("link", { name: "Chat completions" })
    .click();
  await expect(page).toHaveURL(/#\/docs\/api\/chat-completions$/);
  await expect(article(page).getByRole("heading", { level: 1 })).toHaveText(
    "Chat completions",
  );
  await expect(page.getByText("/v1/chat/completions").first()).toBeVisible();
  // a link written as /docs/api/overview inside the page
  await article(page).getByRole("link", { name: "API 總覽" }).first().click();
  await expect(page).toHaveURL(/#\/docs\/api\/overview$/);
  await expect(page.getByTestId("docs-article")).toContainText("API");
  // previous / next at the foot
  await expect(page.getByRole("link", { name: /下一頁/ })).toBeVisible();
});

test("English pages open on a heading anchor and scroll to it", async ({
  page,
}) => {
  await open(page, "docs/api/overview?h=authentication", "en");
  const heading = page.locator("#authentication");
  await expect(heading).toBeVisible();
  await expect
    .poll(async () => (await heading.boundingBox())?.y ?? 9999)
    .toBeLessThan(260);
  await expect(article(page).getByRole("heading", { level: 1 })).toHaveText(
    /API overview|Overview/i,
  );
});

test("tabs switch the example language and code is highlighted lazily", async ({
  page,
}) => {
  await open(page, "docs/api/chat-completions", "en");
  const python = page.getByRole("tab", { name: "Python" }).first();
  await python.click();
  await expect(python).toHaveAttribute("data-state", "active");
  await expect(
    page.getByRole("tabpanel").filter({ hasText: "import" }).first(),
  ).toBeVisible();
});

test("unknown pages say so and link back", async ({ page }) => {
  await open(page, "docs/not/a/page");
  await expect(page.getByText("找不到這份文件")).toBeVisible();
  await page.getByRole("link", { name: "回到文件首頁" }).click();
  await expect(page).toHaveURL(/#\/docs$/);
  await expect(article(page).getByRole("heading", { level: 1 })).toHaveText(
    "簡介",
  );
});

test("the command palette finds docs pages and sections", async ({ page }) => {
  await open(page, "overview", "en");
  await page.getByRole("button", { name: /Search/ }).first().click();
  const input = page.getByPlaceholder("Search pages, models and actions…");
  await input.fill("prompt caching");
  const hit = page
    .getByRole("option")
    .filter({ hasText: /Prompt caching/i })
    .first();
  await expect(hit).toBeVisible();
  await hit.click();
  await expect(page).toHaveURL(/#\/docs\/guides\/prompt-caching/);
  await expect(article(page).getByRole("heading", { level: 1 })).toContainText(
    /Prompt caching/i,
  );
});

test("the language switch re-renders the open page in that language", async ({
  page,
}) => {
  await open(page, "docs/getting-started/install", "en");
  await expect(article(page).getByRole("heading", { level: 1 })).toHaveText(
    /Install/i,
  );
  await page.goto("/console/?lang=zh-CN#/docs/getting-started/install");
  await expect(article(page).getByRole("heading", { level: 1 })).toHaveText(
    /安装/,
  );
});

for (const [hash, label] of [
  ["api", /完整 API 參考/],
  ["settings", /每個設定的說明/],
  ["keys", /驗證與金鑰說明/],
] as const) {
  test(`the ${hash} page links to its docs page`, async ({ page }) => {
    await open(page, hash);
    const link = page.getByTestId("doc-link").first();
    await expect(link).toContainText(label);
    await link.click();
    await expect(page).toHaveURL(/#\/docs\//);
    await expect(article(page)).toBeVisible();
  });
}

test.describe("phone", () => {
  test.use({ viewport: { width: 402, height: 874 } });

  test("pages are chosen from a select and nothing overflows sideways", async ({
    page,
  }) => {
    const errors = await open(page, "docs/api/chat-completions");
    await expect(
      page.getByTestId("docs-nav").getByRole("combobox"),
    ).toBeVisible();
    await expect(
      page.getByTestId("docs-nav").getByRole("link", { name: "Responses" }),
    ).toBeHidden();
    await expect(
      article(page).getByRole("heading", { level: 1 }),
    ).toBeVisible();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - innerWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
    expect(errors).toEqual([]);
  });
});
