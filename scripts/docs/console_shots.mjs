// Console screenshots against a LIVE console and engine. Run only inside gpuq. No route mocks.
// usage: node scripts/docs/console_shots.mjs <base-url> <out-dir>   (needs @playwright/test from frontend/node_modules)
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(join(here, "../../frontend/package.json"));
const { webkit, chromium, devices } = require("@playwright/test");
import { mkdirSync } from "node:fs";
const [base = "http://127.0.0.1:18991", out = "docs/images/console"] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
async function session(eng, opts, scheme) {
  const b = await eng.launch();
  const ctx = await b.newContext({ locale: process.env.LOCALE || "zh-TW", colorScheme: scheme, deviceScaleFactor: 2, ...opts });
  return { b, pg: await ctx.newPage() };
}
const desktop = { viewport: { width: 1440, height: 900 } };
const phone = { ...devices["iPhone 15 Pro"], viewport: { width: 402, height: 874 } };
const pages = process.env.PAGES ? process.env.PAGES.split(",") : ["overview", "requests", "logs", "diagnostics", "models", "downloads", "cache", "playground", "api", "keys", "settings"];
for (const scheme of ["light", "dark"]) {
  const { b, pg } = await session(chromium, desktop, scheme);
  for (const name of pages) {
    await pg.goto(`${base}/console/#/${name}`);
    await pg.waitForTimeout(name === "overview" ? (process.env.FAST ? 6000 : 40000) : 3500);
    if (name === "playground") {
      await pg.locator('[data-testid="playground"] textarea').first().fill("Explain prefix reuse in one sentence.");
      await pg.keyboard.press("Meta+Enter");
      await pg.waitForTimeout(8000);
    }
    await pg.screenshot({ path: `${out}/${name}-${scheme}.png` });
  }
  await pg.goto(`${base}/console/#/overview`); await pg.waitForTimeout(3500);
  await pg.getByTestId("footer-trigger").click(); await pg.waitForTimeout(900);
  await pg.screenshot({ path: `${out}/island-${scheme}.png` });
  await b.close();
}
{
  const { b, pg } = await session(webkit, phone, "dark");
  for (const name of ["overview", "requests"]) {
    await pg.goto(`${base}/console/#/${name}`); await pg.waitForTimeout(name === "overview" ? (process.env.FAST ? 6000 : 40000) : 3500);
    await pg.screenshot({ path: `${out}/mobile-${name}-dark.png` });
  }
  await b.close();
}
console.log("done");
