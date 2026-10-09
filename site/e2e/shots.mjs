// Screenshots for visual review: node e2e/shots.mjs [baseUrl]
// home, an API page, a guide, search, and the language switch; dark + light; 1440 and 402 px.
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";

const BASE = process.argv[2] ?? "http://localhost:3991";
const OUT = new URL("../screenshots/", import.meta.url).pathname;
mkdirSync(OUT, { recursive: true });

const sizes = { desktop: { width: 1440, height: 900 }, phone: { width: 402, height: 874 } };
const themes = ["light", "dark"];
const pages = {
  home: "/zh-TW/",
  api: "/zh-TW/docs/api/chat-completions/",
  guide: "/zh-TW/docs/guides/configuration/",
};

const browser = await chromium.launch();
for (const [sizeName, viewport] of Object.entries(sizes)) {
  for (const theme of themes) {
    const ctx = await browser.newContext({
      viewport,
      colorScheme: theme,
      deviceScaleFactor: sizeName === "phone" ? 2 : 1,
      isMobile: sizeName === "phone",
    });
    await ctx.addInitScript((t) => {
      try {
        localStorage.setItem("theme", t);
      } catch {}
    }, theme);
    const page = await ctx.newPage();
    for (const [name, path] of Object.entries(pages)) {
      await page.goto(BASE + path, { waitUntil: "networkidle" });
      await page.waitForTimeout(400);
      await page.screenshot({ path: `${OUT}${name}-${sizeName}-${theme}.png`, fullPage: name === "home" });
    }
    // search
    await page.goto(BASE + "/zh-TW/docs/api/overview/", { waitUntil: "networkidle" });
    await page.keyboard.press("Control+k");
    await page.waitForTimeout(300);
    if (!(await page.locator("[role=dialog]").count())) {
      await page.locator("button:has-text('搜尋'), [data-search-full], [data-search]").first().click({ timeout: 3000 }).catch(() => {});
    }
    await page.keyboard.type("responses", { delay: 40 });
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `${OUT}search-${sizeName}-${theme}.png` });
    // language switch: open the chooser on the docs page
    await page.goto(BASE + "/zh-TW/docs/getting-started/install/", { waitUntil: "networkidle" });
    if (sizeName === "phone") {
      await page.locator("button[aria-label='Open Sidebar'], button[aria-label*='Sidebar' i]").first().click({ timeout: 3000 }).catch(() => {});
      await page.waitForTimeout(400);
    }
    await page.locator("button[aria-label*='language' i], button[aria-label*='語言'], button[aria-label*='Choose Language' i]").first().click({ timeout: 3000 }).catch(() => {});
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${OUT}lang-${sizeName}-${theme}.png` });
    await ctx.close();
  }
}
await browser.close();
console.log("screenshots in", OUT);
