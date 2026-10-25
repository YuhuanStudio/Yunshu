// Screenshots of the Diagnostics and Requests pages against a live observing engine.
// Usage: node scripts/observe-shots.mjs http://127.0.0.1:18996 OUT_DIR
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";

const [base, out] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1400 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await page.goto(`${base}/console/#/requests`);
await page.getByTestId("observe-panel").waitFor({ timeout: 60000 });
await page
  .getByTestId("observe-row")
  .first()
  .waitFor({ timeout: 120000 })
  .catch(() => {});
await page.waitForTimeout(3000);
await page
  .getByTestId("observe-panel")
  .screenshot({ path: `${out}/requests-debug.png` });
await page.screenshot({ path: `${out}/requests.png` });
await page.goto(`${base}/console/#/diagnostics`);
await page.getByTestId("observe-badge-diag").waitFor({ timeout: 60000 });
await page.waitForTimeout(4000);
await page.screenshot({ path: `${out}/diagnostics.png`, fullPage: true });
console.log(
  JSON.stringify({
    rows: await page
      .getByTestId("observe-row")
      .count()
      .catch(() => 0),
    errors,
  }),
);
await browser.close();
if (errors.length) process.exit(1);
