import { test, type Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { installAsk, installEngine } from "./support/chat-fixture";

// Screenshot tour of the conversation surfaces for design review. Off unless CHAT_SHOTS=<dir>.
const dir = process.env.CHAT_SHOTS;
const theme = process.env.CHAT_THEME === "dark" ? "dark" : "light";
test.skip(!dir, "set CHAT_SHOTS=<dir> to take the tour");

const sizes = {
  desktop: { width: 1440, height: 900 },
  phone: { width: 402, height: 874 },
} as const;

for (const [name, viewport] of Object.entries(sizes)) {
  test.describe(name, () => {
    test.use({
      viewport,
      ...(name === "phone"
        ? { deviceScaleFactor: 3, isMobile: true, hasTouch: true }
        : {}),
    });
    test.beforeEach(async ({ page }) => {
      await page.addInitScript(
        (t) => localStorage.setItem("yunshu.console.theme", t),
        theme,
      );
    });
    const shot = async (page: Page, label: string) => {
      mkdirSync(dir!, { recursive: true });
      await page.screenshot({
        path: `${dir}/${name}-${label}${theme === "dark" ? "-dark" : ""}.png`,
      });
    };

    test("playground", async ({ page, browserName }) => {
      test.skip(browserName !== (name === "phone" ? "webkit" : "chromium"));
      const plan = { current: { words: 90, every: 8, reasoning: "Thinking." } };
      await installEngine(page, plan);
      await page.goto("/console/#/playground");
      await page.getByTestId("playground").waitFor();
      await page.waitForTimeout(600);
      await shot(page, "pg-empty");
      const box = page.getByTestId("playground").locator("textarea");
      for (let i = 0; i < 3; i++) {
        await box.fill(`Question number ${i} about the engine`);
        await box.press("Enter");
        if (i === 2) await page.waitForTimeout(500);
        else
          await page
            .getByRole("button", { name: /停止|Stop/ })
            .waitFor({ state: "detached" });
      }
      await shot(page, "pg-streaming");
      await page
        .getByRole("button", { name: /停止|Stop/ })
        .waitFor({ state: "detached" });
      await shot(page, "pg-done");
      await page.getByRole("tab", { name: /比較|Compare/ }).click();
      await box.fill("Compare these");
      await box.press("Enter");
      await page
        .getByRole("button", { name: /停止|Stop/ })
        .waitFor({ state: "detached" });
      await shot(page, "pg-compare");
    });

    test("docs", async ({ page, browserName }) => {
      test.skip(browserName !== (name === "phone" ? "webkit" : "chromium"));
      await installEngine(page);
      await installAsk(page, "Use **APC**. [1]\n\n```sh\nyunshu serve\n```");
      await page.goto("/console/#/docs");
      await page.getByTestId("docs").waitFor();
      await page.waitForTimeout(800);
      await shot(page, "docs-home");
      const trigger = page.getByTestId("docs-nav").getByRole("combobox");
      if (await trigger.isVisible().catch(() => false)) {
        await trigger.click();
        await page.waitForTimeout(400);
        await shot(page, "docs-dropdown-open");
        await page.keyboard.press("Escape");
      }
      await page
        .getByTestId("docs")
        .getByRole("button", { name: /AI 問答/ })
        .click();
      await page.waitForTimeout(500);
      await shot(page, "ask-empty");
      const input = page.getByTestId("docs-ask").locator("textarea");
      await input.fill("How do I enable APC?");
      await input.press("Enter");
      await page.waitForTimeout(800);
      await shot(page, "ask-answer");
    });
  });
}
