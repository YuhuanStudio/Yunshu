import { test, expect } from "@playwright/test";
import { mkdirSync } from "node:fs";

for (const locale of ["en", "zh-TW", "zh-CN"]) {
  test(`Orama palette retrieves ${locale}`, async ({ page }) => {
    await page.addInitScript(
      (locale) => localStorage.setItem("yunshu.console.locale", locale),
      locale,
    );
    await page.goto("/console/#/docs");
    await expect(
      page.getByTestId("docs-article").getByRole("heading", { level: 1 }),
    ).toBeVisible();
    await page.keyboard.press("Meta+k");
    const input = page.getByRole("dialog").locator("input").first();
    await input.fill(
      locale === "en" ? "prompt caching" : locale === "zh-TW" ? "快取" : "缓存",
    );
    await expect(
      page
        .getByRole("option")
        .filter({ hasText: /Prompt caching|提示詞快取|提示词缓存/ })
        .first(),
    ).toBeVisible();
  });
}

test("OpenAPI try it sends GET models through same origin with auth", async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.locale", "en"),
  );
  let request = "";
  let auth = "";
  await page.route("**/v1/models", async (route) => {
    request = route.request().url();
    auth = route.request().headers().authorization ?? "";
    await route.fulfill({
      json: {
        object: "list",
        data: [{ id: "docsplat-stub", object: "model", owned_by: "yunshu" }],
      },
    });
  });
  await page.goto("/console/#/docs/api/reference");
  await expect(page.getByTestId("api-reference")).toContainText("/v1/models");
  await expect(
    page.getByRole("navigation", { name: "Current location" }),
  ).toContainText("Generated API reference");
  await page.getByRole("button", { name: "Try in Playground" }).click();
  await page
    .getByLabel("Authorization", { exact: true })
    .fill("Bearer docsplat-test-key");
  await page.getByRole("button", { name: /^send/i }).click();
  await expect.poll(() => request).toBe(new URL("/v1/models", page.url()).href);
  expect(auth).toBe("Bearer docsplat-test-key");
  await expect(page.getByRole("dialog")).toContainText("docsplat-stub");
});

for (const width of [1440, 402])
  for (const theme of ["light", "dark"]) {
    test(`docs screenshot ${width} ${theme}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript((theme) => {
        localStorage.setItem("yunshu.console.theme", theme);
        localStorage.setItem("yunshu.console.locale", "en");
      }, theme);
      await page.goto("/console/#/docs/api/chat-completions");
      await expect(
        page.getByTestId("docs-article").getByRole("heading", { level: 1 }),
      ).toBeVisible();
      await expect(
        page.getByRole("tab", { name: "Python" }).first(),
      ).toBeVisible();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth - innerWidth,
        ),
      ).toBeLessThanOrEqual(0);
      mkdirSync("evidence/docsplat", { recursive: true });
      await page.screenshot({
        path: `evidence/docsplat/docs-${width}-${theme}.png`,
        fullPage: true,
      });
    });
  }

test("published Markdown and llms artifacts include the real schema in every locale", async ({
  request,
}) => {
  const index = await request.get("/console/llms.txt");
  expect(index.ok()).toBe(true);
  expect(await index.text()).toContain("/console/docs-assets/openapi.json");
  for (const locale of ["en", "zh-TW", "zh-CN"]) {
    const full = await request.get(`/console/llms-full.${locale}.txt`);
    expect(full.ok()).toBe(true);
    expect(await full.text()).toContain('"/v1/chat/completions"');
    const md = await request.get(
      `/console/docs-assets/${locale}/getting-started/install.md`,
    );
    expect(md.ok()).toBe(true);
    expect(await md.text()).toContain("title:");
  }
});

test("page actions copy source Markdown and expose assistant deep links", async ({
  page,
}) => {
  await page.addInitScript(() => {
    localStorage.setItem("yunshu.console.locale", "en");
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: async (text: string) => {
          (window as Window & { docCopy?: string }).docCopy = text;
        },
      },
    });
  });
  await page.goto("/console/#/docs/getting-started/install");
  await page.getByRole("button", { name: "Copy Markdown" }).click();
  await expect
    .poll(() =>
      page.evaluate(() => (window as Window & { docCopy?: string }).docCopy),
    )
    .toContain("title:");
  for (const name of ["ChatGPT", "Claude"]) {
    const link = page
      .getByTestId("docs-article")
      .getByRole("link", { name, exact: true });
    await expect(link).toHaveAttribute(
      "href",
      /docs-assets%2Fen%2Fgetting-started%2Finstall.md/,
    );
  }
});
