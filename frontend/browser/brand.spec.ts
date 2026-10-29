import { expect, test, type Page } from "@playwright/test";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { installAsk, installEngine } from "./support/chat-fixture";

// The Yunshu brand: one real asset set (favicon, touch icon, manifest) and the same tile on every surface,
// with the sidebar's brand block exactly as YunUI's Sidebar draws it for Yunxin (mark + name as ONE link to
// home, 36 px tile, 18 px semibold name, hover wash, collapse button after a hairline). BRAND_SHOTS=/dir
// writes the screenshots.

const SHOTS = process.env.BRAND_SHOTS;
if (SHOTS) mkdirSync(SHOTS, { recursive: true });
const shot = async (page: Page, name: string, clip?: object) => {
  if (SHOTS)
    await page.screenshot({
      path: `${SHOTS}/${name}.png`,
      ...(clip ? { clip: clip as never } : {}),
    });
};

test.use({ viewport: { width: 1440, height: 900 } });

const open = async (page: Page, theme: "light" | "dark" = "light") => {
  await installEngine(page);
  await page.addInitScript((t) => {
    localStorage.setItem("yunshu.console.url", location.origin);
    localStorage.setItem("yunshu.console.theme", t);
  }, theme);
  await page.goto("/console/#/overview");
  await page.getByRole("navigation").first().waitFor();
};
const rail = (page: Page) => page.getByRole("navigation").first();
const block = (page: Page) => rail(page).locator("a").first();

test("the asset set is declared and every file resolves", async ({
  page,
  request,
}) => {
  await open(page);
  const head = await page.evaluate(() => ({
    icons: [...document.querySelectorAll('link[rel~="icon"]')].map((l) => [
      l.getAttribute("href"),
      l.getAttribute("type") ?? l.getAttribute("sizes"),
    ]),
    touch: document
      .querySelector('link[rel="apple-touch-icon"]')
      ?.getAttribute("href"),
    manifest: document
      .querySelector('link[rel="manifest"]')
      ?.getAttribute("href"),
    themes: [...document.querySelectorAll('meta[name="theme-color"]')].map(
      (m) => m.getAttribute("media"),
    ),
  }));
  expect(head.icons.map((i) => i[0])).toEqual([
    "/console/favicon.svg",
    "/console/favicon.ico",
  ]);
  expect(head.touch).toBe("/console/apple-touch-icon.png");
  expect(head.manifest).toBe("/console/manifest.json");
  expect(head.themes).toHaveLength(2);
  for (const href of [
    ...head.icons.map((i) => i[0]!),
    head.touch!,
    head.manifest!,
  ]) {
    const r = await request.get(href);
    expect(r.status(), href).toBe(200);
    expect(r.headers()["content-type"], href).toMatch(/image|json/);
  }
  const manifest = await (await request.get(head.manifest!)).json();
  for (const icon of manifest.icons)
    expect((await request.get(icon.src)).status(), icon.src).toBe(200);
});

test("the tab icon is the tile, and the theme switch moves the browser theme-color", async ({
  page,
}) => {
  await open(page);
  const color = () =>
    page.evaluate(() =>
      [...document.querySelectorAll('meta[name="theme-color"]')].map(
        (m) => (m as HTMLMetaElement).content,
      ),
    );
  expect(await color()).toEqual(["#f7f7f8", "#f7f7f8"]);
  await page.getByRole("button", { name: /深色|dark/i }).click();
  await expect.poll(color).toEqual(["#0b0b0d", "#0b0b0d"]);
  // the icon a browser tab shows: render the svg at 16 px next to the ico's smallest entry
  if (SHOTS) {
    await page.setContent(
      `<body style="margin:0;padding:12px;display:flex;gap:10px;align-items:center;background:#e4e4e7;font:12px system-ui">
        <span style="display:inline-flex;align-items:center;gap:8px;background:#fff;border-radius:8px 8px 0 0;padding:8px 12px;width:210px"><img src="/console/favicon.svg" width=16 height=16>預填 46% · 雲樞</span>
        <span style="display:inline-flex;align-items:center;gap:8px;background:#27272a;color:#fafafa;border-radius:8px 8px 0 0;padding:8px 12px;width:210px"><img src="/console/favicon.ico" width=16 height=16>運作中 · 雲樞</span></body>`,
    );
    await page.waitForTimeout(200);
    await shot(page, "tab-favicon", { x: 0, y: 0, width: 480, height: 56 });
  }
});

for (const theme of ["light", "dark"] as const) {
  test(`sidebar brand block (${theme}): one link, Yunxin's sizes, hover, collapse`, async ({
    page,
  }) => {
    await open(page, theme);
    const link = block(page);
    // one link holds the mark and the name
    await expect(link).toHaveAttribute("href", "#/overview");
    await expect(link.locator("img")).toHaveAttribute(
      "src",
      "/console/favicon.svg",
    );
    await expect(link).toContainText("Yunshu");
    const m = await page.evaluate(() => {
      const nav = document.querySelector("nav")!;
      const a = nav.querySelector("a")!;
      const img = a.querySelector("img")!;
      const name = a.querySelector("span")!;
      const row = a.parentElement!;
      const btn = row.querySelector("button")!;
      const cs = (e: Element) => getComputedStyle(e);
      const r = (e: Element) => e.getBoundingClientRect();
      return {
        img: [r(img).width, r(img).height],
        font: [cs(name).fontSize, cs(name).fontWeight],
        padLeft: cs(a).paddingLeft,
        padY: [cs(a).paddingTop, cs(a).paddingBottom],
        radius: cs(a).borderTopLeftRadius,
        gap: cs(a).columnGap,
        cursor: cs(a).cursor,
        rowGap: cs(row).columnGap,
        btnAfterLink: r(btn).left >= r(a).right,
        btnLabel: btn.getAttribute("aria-label"),
        linkRight: r(a).right,
        railRight: r(nav).right,
        railWidth: r(nav).width,
      };
    });
    expect(m.img).toEqual([36, 36]);
    expect(m.font).toEqual(["18px", "600"]);
    expect(m.padLeft).toBe("12px");
    expect(m.padY).toEqual(["6px", "6px"]);
    expect(m.radius).toBe("8px");
    expect(m.gap).toBe("10px");
    expect(m.rowGap).toBe("8px");
    expect(["pointer", "auto"]).toContain(m.cursor); // WebKit reports a link's UA cursor as auto
    expect(m.btnAfterLink).toBe(true);
    expect(m.btnLabel).toBeTruthy();
    expect(m.railWidth).toBe(256);
    await shot(page, `sidebar-expanded-${theme}`, {
      x: 0,
      y: 0,
      width: 256,
      height: 180,
    });
    // hover: the wash appears on the link (one hover surface for mark and name)
    const bg = () => link.evaluate((e) => getComputedStyle(e).backgroundColor);
    const rest = await bg();
    await link.hover();
    await expect.poll(bg).not.toBe(rest);
    await shot(page, `sidebar-hover-${theme}`, {
      x: 0,
      y: 0,
      width: 256,
      height: 90,
    });
    // keyboard focus: the link shows a ring
    await page.mouse.move(700, 500);
    // a key press first, so the focus that follows counts as keyboard focus (WebKit's Tab skips links)
    await page.keyboard.press("Shift");
    await link.focus();
    await expect(link).toBeFocused();
    const ring = await link.evaluate((e) => {
      const s = getComputedStyle(e);
      return s.outlineStyle !== "none" || s.boxShadow !== "none";
    });
    expect(ring, "focus ring on the brand link").toBe(true);
    await shot(page, `sidebar-focus-${theme}`, {
      x: 0,
      y: 0,
      width: 256,
      height: 90,
    });
    // clicking it goes home
    await page.evaluate(() => (location.hash = "#/requests"));
    await expect(page).toHaveURL(/#\/requests/);
    await link.click();
    await expect(page).toHaveURL(/#\/overview/);
  });
}

test("collapse: the whole rail leaves (as in Yunxin) and a button in the top bar brings it back", async ({
  page,
}) => {
  await open(page);
  const nav = rail(page);
  const collapse = nav.getByRole("button").first();
  await collapse.click();
  await expect
    .poll(() => nav.evaluate((e) => e.getBoundingClientRect().right))
    .toBeLessThanOrEqual(0);
  await shot(page, "sidebar-collapsed-light", {
    x: 0,
    y: 0,
    width: 640,
    height: 120,
  });
  await page.getByRole("button", { name: /展開|Expand/ }).click();
  await expect
    .poll(() => nav.evaluate((e) => e.getBoundingClientRect().left))
    .toBe(0);
});

test("the same tile marks the docs title, the Ask AI dialog and the assistant's avatar", async ({
  page,
}) => {
  await installEngine(page);
  await installAsk(page);
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.goto("/console/#/docs");
  const docs = page.getByTestId("docs");
  const title = docs.locator('img[src="/console/favicon.svg"]:visible').first();
  await expect(title).toBeVisible();
  await shot(page, "docs-nav", { x: 256, y: 0, width: 1184, height: 200 });
  await docs.getByRole("button", { name: /AI 問答/ }).click();
  const dialog = page.getByTestId("docs-ask");
  await expect(
    dialog.locator('img[src="/console/favicon.svg"]').first(),
  ).toBeVisible();
  const field = dialog.locator("textarea");
  await field.fill("How do I enable APC?");
  await field.press("Enter");
  await expect(dialog.getByTestId("docs-ask-answer")).toContainText("APC");
  // title + empty state were replaced by the turn: the avatar is the tile too
  await expect(
    dialog.locator('img[src="/console/favicon.svg"]').first(),
  ).toBeVisible();
  expect(
    await dialog.locator("svg.lucide-bot, svg.lucide-sparkles").count(),
  ).toBe(0);
  await shot(page, "ask-ai-light");
});

test("the Playground's assistant avatar is the tile", async ({ page }) => {
  await installEngine(page, { current: { words: 12, every: 5 } });
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.goto("/console/#/playground");
  const input = page
    .getByTestId("playground")
    .locator('[data-yunui="chat-composer"] textarea');
  await input.fill("Hello");
  await input.press("Enter");
  await page
    .getByRole("button", { name: /停止|Stop/ })
    .waitFor({ state: "detached" });
  await expect(
    page
      .getByTestId("playground")
      .locator('[data-yunui="chat-message"] img[src="/console/favicon.svg"]')
      .first(),
  ).toBeVisible();
  await shot(page, "chat-avatar-light");
});

// Picture: Yunshu's block next to Yunxin's. Both are YunUI Sidebar's own brand block (the same markup and
// classes, see the measurements above); the Yunxin one is that markup with Yunxin's favicon and name.
test("brand block next to Yunxin's (picture only)", async ({ page }) => {
  test.skip(!SHOTS, "set BRAND_SHOTS to write the picture");
  const yunxin =
    "/Users/yuhuan/Documents/YuhuanStudio/Yunxin/frontend/public/favicon.ico";
  test.skip(!existsSync(yunxin), "Yunxin checkout not present");
  const icon = `data:image/x-icon;base64,${readFileSync(yunxin).toString("base64")}`;
  for (const theme of ["light", "dark"] as const) {
    await open(page, theme);
    const html = await rail(page)
      .locator("> div")
      .first()
      .evaluate((e) => e.outerHTML);
    await page.evaluate(
      ({ html, icon }) => {
        const wrap = document.createElement("div");
        wrap.id = "yunxin-compare";
        wrap.style.cssText =
          "position:fixed;left:300px;top:20px;width:256px;z-index:99;background:var(--bg-window);border:1px solid var(--border-hairline);border-radius:8px";
        wrap.innerHTML = html;
        const img = wrap.querySelector("img")!;
        img.src = icon;
        img.alt = "Yunxin";
        img.style.borderRadius = "8px";
        wrap.querySelector("span")!.textContent = "Yunxin";
        document.body.appendChild(wrap);
      },
      { html, icon },
    );
    await shot(page, `compare-yunxin-${theme}`, {
      x: 0,
      y: 0,
      width: 580,
      height: 90,
    });
  }
});
