// Screenshots + visible text of the console's Overview and Requests pages every few seconds while a real
// engine is under load (the agent-load reproducer starts this next to the scenario).
//
//   node scripts/agentload-shots.mjs --base http://127.0.0.1:18997 --out DIR [--seconds 600] [--interval 5]
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "@playwright/test";

const { values } = parseArgs({
  options: {
    base: { type: "string" },
    out: { type: "string" },
    seconds: { type: "string", default: "600" },
    interval: { type: "string", default: "5" },
  },
});
if (!values.base || !values.out) throw new Error("--base and --out are required");
mkdirSync(values.out, { recursive: true });
const browser = await chromium.launch();
const pages = {};
for (const name of ["overview", "requests", "diagnostics"]) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await ctx.newPage();
  await page.goto(`${values.base}/console/#/${name}`, { waitUntil: "domcontentloaded" });
  pages[name] = page;
}
const t0 = Date.now();
const dump = [];
let stop = false;
process.on("SIGTERM", () => (stop = true));
process.on("SIGINT", () => (stop = true));
for (let i = 0; !stop && Date.now() - t0 < Number(values.seconds) * 1000; i++) {
  const t = Math.round((Date.now() - t0) / 1000);
  const row = { t, pages: {} };
  for (const [name, page] of Object.entries(pages)) {
    try {
      if (name === "requests") {
        const tab = page.getByRole("tab", { name: "進行中", exact: true });
        if (await tab.count()) await tab.first().click({ timeout: 1000 }).catch(() => {});
      }
      await page.screenshot({ path: join(values.out, `${name}-${String(i).padStart(4, "0")}-t${t}.png`) });
      row.pages[name] = (await page.locator("main").innerText({ timeout: 2000 })).slice(0, 6000);
    } catch (e) {
      row.pages[name] = `ERR ${String(e).slice(0, 120)}`;
    }
  }
  dump.push(row);
  writeFileSync(join(values.out, "text.json"), JSON.stringify(dump, null, 1));
  await new Promise((r) => setTimeout(r, Number(values.interval) * 1000));
}
await browser.close();
