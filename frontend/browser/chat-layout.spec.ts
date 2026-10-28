import { expect, test, type Page } from "@playwright/test";
import { installAsk, installEngine, setPlan } from "./support/chat-fixture";

// The conversation surfaces (Playground and Ask AI) share one layout: a transcript that scrolls and sticks
// to the bottom, one column for messages and composer, and a composer pinned under it. These specs fail on
// the earlier hand-assembled layouts (header taking a third of a phone, composer narrower than its row, ...).

const PHONE = { width: 402, height: 874 };
const DESKTOP = { width: 1440, height: 900 };
const stop = (page: Page) => page.getByRole("button", { name: /停止|Stop/ });

const box = async (page: Page, selector: string) => {
  const b = await page.locator(selector).first().boundingBox();
  if (!b) throw new Error(`no box for ${selector}`);
  return b;
};
const scroller = (page: Page) =>
  page.locator('[data-scroll-container="true"]').first();

async function openPlayground(page: Page, plan = {}) {
  const planned = {
    current: { words: 30, every: 5, reasoning: "Thinking.", ...plan },
  };
  await installEngine(page, planned);
  await page.goto("/console/#/playground");
  await page.getByTestId("playground").waitFor();
  return page.getByTestId("playground").locator("textarea");
}
async function ask(page: Page, input: ReturnType<Page["locator"]>, q: string) {
  await input.fill(q);
  await input.press("Enter");
}
const noOverflow = async (page: Page) =>
  expect(
    await page.evaluate(() => ({
      doc: document.documentElement.scrollWidth - innerWidth,
      list:
        (
          document.querySelector(
            '[data-scroll-container="true"]',
          ) as HTMLElement
        )?.scrollWidth -
        (
          document.querySelector(
            '[data-scroll-container="true"]',
          ) as HTMLElement
        )?.clientWidth,
    })),
  ).toEqual({ doc: 0, list: 0 });

test.describe("phone", () => {
  test.use({
    viewport: PHONE,
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
  });

  test("playground: the transcript owns most of the screen and nothing overflows sideways", async ({
    page,
    browserName,
  }) => {
    test.skip(browserName !== "webkit");
    const input = await openPlayground(page, {
      words: 60,
      tail:
        "\n\n" + "x".repeat(400) + "\n\n```sh\n" + "y".repeat(300) + "\n```",
    });
    await page.waitForTimeout(500);
    const list = await box(page, '[data-scroll-container="true"]');
    // before: the title block, model, API format, mode and five wrapped buttons took 35 % of the phone.
    expect(list.height).toBeGreaterThan(PHONE.height * 0.5);
    const toolbar = await box(page, '[data-testid="playground-toolbar"]');
    expect(toolbar.height).toBeLessThan(110);
    await ask(page, input, "Hello");
    await stop(page).waitFor({ state: "detached" });
    await noOverflow(page);
    const composer = await box(page, "textarea");
    expect(composer.y + composer.height).toBeLessThanOrEqual(PHONE.height);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollHeight - innerHeight,
      ),
    ).toBe(0);
  });

  test("ask AI: full-width composer and no horizontal overflow", async ({
    page,
    browserName,
  }) => {
    test.skip(browserName !== "webkit");
    await installEngine(page);
    await installAsk(page, "Use **APC**. " + "z".repeat(300));
    await page.goto("/console/#/docs");
    await page
      .getByTestId("docs")
      .getByRole("button", { name: /AI 問答/ })
      .click();
    const dialog = page.getByTestId("docs-ask");
    await expect(dialog).toBeVisible();
    const field = dialog.locator("textarea");
    const d = (await dialog.boundingBox())!;
    const f = (await field.boundingBox())!;
    // before: the textarea was a ~210 px box next to a floating send button.
    expect(f.width).toBeGreaterThan(d.width * 0.7);
    await ask(page, field, "How do I enable APC?");
    await expect(dialog.getByTestId("docs-ask-answer")).toContainText("APC");
    await noOverflow(page);
    expect(d.height).toBeGreaterThan(PHONE.height * 0.9);
  });
});

test.describe("desktop", () => {
  test.use({ viewport: DESKTOP });

  test("playground: messages and composer share one column; the composer stays put as the reply grows", async ({
    page,
    browserName,
  }) => {
    test.skip(browserName !== "chromium");
    const input = await openPlayground(page, { words: 200, every: 6 });
    const before = await box(page, "textarea");
    await ask(page, input, "Explain the KV cache");
    await page.getByText("word20").first().waitFor();
    const mid = await box(page, "textarea");
    await stop(page).waitFor({ state: "detached" });
    const after = await box(page, "textarea");
    expect(Math.abs(mid.y - before.y)).toBeLessThan(2);
    expect(Math.abs(after.y - before.y)).toBeLessThan(2);
    const bubble = await box(
      page,
      '[data-yunui="chat-message"][data-role="user"] > div',
    );
    const composer = await box(page, '[data-yunui="chat-composer"]');
    // before: the bubble hugged the right edge of a wider wrapper than the composer.
    expect(
      Math.abs(bubble.x + bubble.width - (composer.x + composer.width)),
    ).toBeLessThan(2);
    await noOverflow(page);
  });

  test("playground: following the stream stops when the reader scrolls up, and the pill brings it back", async ({
    page,
    browserName,
  }) => {
    test.skip(browserName !== "chromium");
    const input = await openPlayground(page, { words: 500, every: 12 });
    await ask(page, input, "Write a lot");
    const list = scroller(page);
    await expect
      .poll(() =>
        list.evaluate((el) => el.scrollHeight - el.clientHeight > 200),
      )
      .toBe(true);
    const atBottom = () =>
      list.evaluate(
        (el) => el.scrollHeight - el.scrollTop - el.clientHeight < 8,
      );
    await expect.poll(atBottom).toBe(true);
    await list.hover();
    await page.mouse.wheel(0, -600);
    const top = await list.evaluate((el) => el.scrollTop);
    await page.waitForTimeout(600);
    expect(await list.evaluate((el) => el.scrollTop)).toBeLessThanOrEqual(
      top + 2,
    );
    const pill = page.locator('[data-yunui="jump-to-latest"]');
    await expect(pill).toHaveAttribute("data-visible", "true");
    await pill.click();
    await expect.poll(atBottom).toBe(true);
    await stop(page)
      .click({ timeout: 1000 })
      .catch(() => {});
  });

  test("playground: a finished reply can be copied and regenerated", async ({
    page,
    browserName,
  }) => {
    test.skip(browserName !== "chromium");
    const input = await openPlayground(page, { words: 12 });
    let calls = 0;
    page.on("request", (r) => {
      if (r.url().endsWith("/v1/chat/completions")) calls++;
    });
    await ask(page, input, "Hi");
    await stop(page).waitFor({ state: "detached" });
    await expect(page.getByRole("button", { name: "複製回覆" })).toBeVisible();
    await setPlan(page, { words: 8, every: 5 });
    await page.getByRole("button", { name: "重新生成" }).click();
    await stop(page).waitFor({ state: "detached" });
    await expect(page.getByText("word7")).toBeVisible();
    await expect(page.locator('[data-role="user"]')).toHaveCount(1);
    await expect(page.locator('[data-role="assistant"]')).toHaveCount(1);
  });

  test("ask AI: Enter sends, Shift+Enter breaks the line, and the composer is the full dialog width", async ({
    page,
    browserName,
  }) => {
    test.skip(browserName !== "chromium");
    await installEngine(page);
    await installAsk(page);
    await page.goto("/console/#/docs");
    await page
      .getByTestId("docs")
      .getByRole("button", { name: /AI 問答/ })
      .click();
    const dialog = page.getByTestId("docs-ask");
    const field = dialog.locator("textarea");
    const d = (await dialog.boundingBox())!;
    expect((await field.boundingBox())!.width).toBeGreaterThan(d.width * 0.7);
    await field.fill("line one");
    await field.press("Shift+Enter");
    await field.type("line two");
    await expect(field).toHaveValue("line one\nline two");
    await field.press("Enter");
    await expect(dialog.getByTestId("docs-ask-answer")).toContainText("APC");
    await expect(field).toHaveValue("");
  });
});
