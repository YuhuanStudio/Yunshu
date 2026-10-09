// Console screenshots for the docs: the Vite dev console against Playwright route mocks (no engine, no GPU).
// usage: node scripts/docs/console_shots.mjs <base-url> <out-dir>   (needs @playwright/test from frontend/node_modules)
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(join(here, "../../frontend/package.json"));
const { webkit, chromium, devices } = require("@playwright/test");
import { install, statusBody } from "./console_mock.mjs";

const [base = "http://127.0.0.1:18991", out = "docs/images/console"] = process.argv.slice(2);
const MODEL = "Qwen3.8-27B-oQ4e-mtp";
const decode = () => statusBody({
  requests: { active: 2, queued: 0, prefill: 1, decode: 1, items: [
    { request_id: "req_a1b2c3d4", elapsed_s: 9, phase: "decode", model: MODEL, completion_tokens: 212, tokens_per_second: 61.3 },
    { request_id: "req_e5f6a7b8", elapsed_s: 3.2, phase: "prefill", model: MODEL, prompt_tokens: 12000, cached_tokens: 8000, processed_tokens: 9400, percent: 35, tokens_per_second: 880, eta_s: 4.1 },
  ] },
  throughput: { window_s: 60, requests: 6, prompt_tokens: 38000, completion_tokens: 1450, live_decode_tps: 61.3, mean_prefill_tps: 900, mean_decode_tps: 58.4 },
});
const reply = "Prefix reuse means the engine keeps the key/value state of a prompt it has already processed, so a later request that starts with the same tokens skips that prefill work and only computes the new suffix.";
const chunks = reply.match(/.{1,24}/g);
const sse = chunks.map((c, i) => `data: ${JSON.stringify({ choices: [{ delta: { content: c }, finish_reason: i === chunks.length - 1 ? "stop" : null }], ...(i === chunks.length - 1 ? { usage: { prompt_tokens: 18, completion_tokens: 46 } } : {}) })}\n\n`).join("") + "data: [DONE]\n\n";

async function session(eng, opts, scheme) {
  const b = await eng.launch();
  const ctx = await b.newContext({ locale: process.env.LOCALE || "zh-TW", colorScheme: scheme, deviceScaleFactor: 2, ...opts });
  await ctx.addInitScript(() => localStorage.setItem("yunshu.console.url", location.origin));
  await install(ctx, { status: decode });
  await ctx.route("**/debug/**", (r) => {
    const path = new URL(r.request().url()).pathname;
    const body = path === "/debug/kv-cache" ? { caches: [{ model_id: "Qwen3.8-27B-oQ4e-mtp", apc: { entries: 14, resident_bytes: 2.4e9, warm_bytes: 0, warm_ratio: 1, disk_bytes: 1.2e10, lookups_hit: 95, lookups_miss: 25, matched_tokens: 640000, memory_evictions: 2, memory_skips: 0, warm_demotions: 0 } }] } : path === "/debug/system" ? { cpu: { percent: 14, logical_cores: 16 }, memory: { percent: 41, used_bytes: 5.6e10, total_bytes: 1.37e11 }, gpu: { active_bytes: 2.5e10 } } : {};
    return r.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  await ctx.route("**/v1/chat/completions", (r) => r.fulfill({ status: 200, contentType: "text/event-stream", body: sse }));
  return { b, pg: await ctx.newPage() };
}
const desktop = { viewport: { width: 1440, height: 900 } };
const phone = { ...devices["iPhone 15 Pro"], viewport: { width: 402, height: 874 } };
const pages = process.env.PAGES ? process.env.PAGES.split(",") : ["overview", "requests", "logs", "diagnostics", "models", "downloads", "cache", "playground", "api", "keys", "settings"];
for (const scheme of ["light", "dark"]) {
  const { b, pg } = await session(chromium, desktop, scheme);
  for (const name of pages) {
    await pg.goto(`${base}/console/#/${name}`);
    await pg.waitForTimeout(name === "overview" ? 18000 : 3500);
    if (name === "playground") {
      await pg.addStyleTag({ content: '[data-testid="reply-stats"]{display:none}' });
      await pg.locator('[data-testid="playground"] textarea').first().fill("Explain prefix reuse in one sentence.");
      await pg.keyboard.press("Meta+Enter").catch(() => {});
      await pg.waitForTimeout(1500);
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
    await pg.goto(`${base}/console/#/${name}`); await pg.waitForTimeout(name === "overview" ? 18000 : 3500);
    await pg.screenshot({ path: `${out}/mobile-${name}-dark.png` });
  }
  await b.close();
}
console.log("done");
