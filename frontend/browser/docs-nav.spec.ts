import { expect, test, type Page } from "@playwright/test";
import { installEngine } from "./support/chat-fixture";

// Docs page navigation on a phone and on a desktop: the page tree has 70+ pages, the last one must be
// reachable, the open navigation must fit the screen and scroll, and nothing may overflow sideways.

const LAST = "路線圖與提案"; // developers/roadmap, the last page of the tree (zh-TW)
const NAV_TRIGGER = /跳到文件頁面|開啟文件目錄/;

async function open(page: Page, hash = "docs") {
  await installEngine(page);
  await page.goto(`/console/#/${hash}`, { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("docs-article")).toBeVisible();
}

const trigger = (page: Page) =>
  page
    .getByRole("combobox", { name: NAV_TRIGGER })
    .or(page.getByRole("button", { name: NAV_TRIGGER }));
// The open navigation: the drawer on a phone (the popup before the page tree became fumadocs').
const panel = (page: Page) =>
  page.locator('#nd-sidebar-mobile, [role="listbox"]');
const entry = (page: Page, name: string) =>
  panel(page)
    .getByRole("option", { name, exact: true })
    .or(panel(page).getByRole("link", { name, exact: true }));

test.describe("phone", () => {
  test.use({
    viewport: { width: 402, height: 874 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
  });
  test.beforeEach(({ browserName }) => test.skip(browserName !== "webkit"));

  test("the open page navigation fits the screen and reaches the last page", async ({
    page,
  }) => {
    await open(page);
    await trigger(page).first().tap();
    const last = entry(page, LAST).first();
    await last.scrollIntoViewIfNeeded({ timeout: 5000 });
    await expect(last).toBeInViewport();
    // The panel never grows past the screen (it was a 3070 px popup that could not scroll).
    const tallest = await page.evaluate(() =>
      Math.max(
        ...[
          ...document.querySelectorAll<HTMLElement>(
            '[role="listbox"], [role="dialog"], [data-state="open"]',
          ),
        ].map((el) => el.getBoundingClientRect().height),
      ),
    );
    expect(tallest).toBeLessThanOrEqual(874);
    await last.tap();
    await expect(page).toHaveURL(/#\/docs\/developers\/roadmap$/);
    await expect(
      page.getByTestId("docs-article").getByRole("heading", { level: 1 }),
    ).toHaveText(LAST);
    // Choosing a page closes the navigation.
    await expect(entry(page, LAST).first()).toBeHidden();
  });

  test("Escape closes the page navigation", async ({ page }) => {
    await open(page);
    await trigger(page).first().tap();
    await expect(entry(page, "安裝").first()).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(entry(page, "安裝").first()).toBeHidden();
  });

  for (const hash of ["docs", "docs/api/chat-completions"]) {
    test(`no horizontal overflow on ${hash}`, async ({ page }) => {
      await open(page, hash);
      await page.waitForTimeout(500);
      const over = await page.evaluate(() => {
        const scroller = document.querySelector('[data-testid="page-scroll"]')!;
        return {
          doc: document.documentElement.scrollWidth - innerWidth,
          scroller: scroller.scrollWidth - scroller.clientWidth,
        };
      });
      expect(over.doc).toBeLessThanOrEqual(0);
      expect(over.scroller).toBeLessThanOrEqual(0);
    });
  }
});

test.describe("desktop", () => {
  test.use({ viewport: { width: 1440, height: 900 } });
  test.beforeEach(({ browserName }) => test.skip(browserName !== "chromium"));

  test("the page tree stays in view while the page scrolls and links navigate", async ({
    page,
  }) => {
    await open(page, "docs/api/chat-completions");
    const nav = page.getByTestId("docs-nav");
    await expect(nav.getByRole("link", { name: "Responses" })).toBeVisible();
    await page
      .getByTestId("page-scroll")
      .evaluate((el) => el.scrollTo(0, 1200));
    await expect(nav.getByRole("link", { name: "Responses" })).toBeInViewport();
    await nav.getByRole("link", { name: "Responses" }).click();
    await expect(page).toHaveURL(/#\/docs\/api\/responses$/);
  });

  test("the last page of the tree is reachable from the sidebar", async ({
    page,
  }) => {
    await open(page);
    const link = page.getByTestId("docs-nav").getByRole("link", { name: LAST });
    await link.scrollIntoViewIfNeeded({ timeout: 5000 });
    await expect(link).toBeInViewport();
    await link.click();
    await expect(page).toHaveURL(/#\/docs\/developers\/roadmap$/);
  });

  test("the on-this-page links jump to their heading without leaving the page", async ({
    page,
  }) => {
    await open(page, "docs/api/chat-completions");
    const toc = page.getByRole("complementary", { name: /本頁內容/ });
    const links = toc.getByRole("link");
    await expect(links.nth(1)).toBeVisible();
    const target = links.nth(1);
    const id = (await target.getAttribute("href"))!.slice(1);
    await target.click();
    await expect(page).toHaveURL(/#\/docs\/api\/chat-completions\?h=/);
    await expect(page.locator(`[id="${id}"]`)).toBeInViewport();
  });
});

test.describe("API reference operation picker on a phone", () => {
  test.use({
    viewport: { width: 402, height: 874 },
    isMobile: true,
    hasTouch: true,
  });
  test("the 177-operation list is capped to the screen, scrolls, and an operation far down can be picked", async ({
    page,
    browserName,
  }) => {
    test.skip(browserName !== "webkit");
    await installEngine(page);
    await page.goto("/console/#/docs/api/reference");
    await page.getByRole("combobox").first().tap();
    const list = page.locator("[role=listbox]");
    await expect(list).toBeVisible();
    const box = (await list.boundingBox())!;
    expect(box.height).toBeLessThanOrEqual(874);
    expect(box.x + box.width).toBeLessThanOrEqual(402);
    const far = page.locator("[role=option]").nth(120);
    await far.scrollIntoViewIfNeeded();
    await far.tap();
    await expect(page).toHaveURL(/reference\?operation=/);
  });
});
