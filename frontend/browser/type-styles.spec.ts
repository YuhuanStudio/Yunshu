import { expect, test, type Page, type Route } from "@playwright/test";

/**
 * RULES: at most 8 type styles app-wide on the YunDesign scale (13 body, 12 label, 14/600 section,
 * 20 page title, 22 stat value, plus mono for ids and code). A style is size / weight / mono-or-sans,
 * measured on every rendered text node of every page.
 */
const MAX_STYLES = 8;
/** The YunDesign scale: sizes 12, 13, 14, 20 and 22 only, weights 400-600. */
const SCALE_SIZES = new Set(["12px", "13px", "14px", "20px", "22px"]);
const PAGES = [
  "overview",
  "requests",
  "logs",
  "diagnostics",
  "models",
  "downloads",
  "cache",
  "playground",
  "api",
  "keys",
  "settings",
  "models/Qwen3.8-27B-oQ4e-mtp",
];

const status = {
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "running",
  uptime_s: 4000,
  load_error: null,
  models: [
    {
      id: "Qwen3.8-27B-oQ4e-mtp",
      type: "VLM",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 17.6,
    },
    {
      id: "Qwen3.5-9B",
      type: "LLM",
      loaded: false,
      loading: false,
      pinned: false,
      size_gb: 5.8,
    },
  ],
  memory: { active_gb: 27.1, cache_gb: 2, peak_gb: 30, total_gb: 137 },
  requests: { active: 0, queued: 0, prefill: 0, decode: 0, items: [] },
  last: {
    request_id: "req-9",
    prompt_tokens: 4096,
    completion_tokens: 600,
    cached_tokens: 3900,
    prefill_tps: 190,
    decode_tps: 41.2,
    ttft_ms: 320,
    t: 1.79e9,
  },
  throughput: {
    window_s: 60,
    requests: 3,
    prompt_tokens: 100,
    completion_tokens: 50,
    live_decode_tps: null,
    mean_prefill_tps: 98.9,
    mean_decode_tps: 41.9,
  },
};

async function install(page: Page) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route: Route) => {
    const path = new URL(route.request().url()).pathname;
    const json = (s: number, b: unknown) =>
      route.fulfill({
        status: s,
        contentType: "application/json",
        body: JSON.stringify(b),
      });
    if (path === "/v1/yunshu/status") return json(200, status);
    if (path === "/v1/models")
      return json(200, {
        object: "list",
        data: status.models.map((m) => ({ id: m.id })),
      });
    if (path === "/v1/yunshu/keys")
      return json(200, {
        data: [
          {
            id: "key_1",
            name: "laptop",
            prefix: "ys-abcd",
            scopes: ["infer"],
            created: 1.79e9,
            last_used: null,
            disabled: false,
            quotas: {},
            expires: null,
          },
        ],
      });
    return json(404, { detail: "fixture endpoint not found" });
  });
}

/** Distinct "size/weight/family" combinations over every visible text node. */
export async function typeStyles(page: Page): Promise<Record<string, string>> {
  return page.evaluate(() => {
    const out: Record<string, string> = {};
    const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let n: Node | null;
    while ((n = w.nextNode())) {
      const text = (n.textContent ?? "").trim();
      const el = n.parentElement;
      if (!text || !el || el.closest("script,style,svg,.sr-only")) continue;
      const cs = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      if (cs.display === "none" || cs.visibility === "hidden" || r.width < 3)
        continue;
      const mono = /mono|menlo|courier|consolas/i.test(cs.fontFamily);
      const key = `${cs.fontSize}/${cs.fontWeight}/${mono ? "mono" : "sans"}`;
      out[key] ??= `${el.tagName.toLowerCase()}.${String(el.className).split(" ").slice(0, 3).join(".")} "${text.slice(0, 24)}"`;
    }
    return out;
  });
}

test("the whole console uses at most 8 type styles", async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await install(page);
  const all: Record<string, string> = {};
  for (const p of PAGES) {
    await page.goto(`/console/#/${p}`, { waitUntil: "domcontentloaded" });
    await page.reload({ waitUntil: "domcontentloaded" });
    await page.waitForTimeout(1500);
    const found = await typeStyles(page);
    if (process.env.TYPE_STYLES_LOG)
      console.log(p, Object.keys(found).length, (await page.locator("body").innerText()).length, (await page.locator("body").innerText()).slice(0, 200).replace(/\n/g, "|"));
    for (const [k, v] of Object.entries(found)) all[k] ??= `${p}: ${v}`;
  }
  const keys = Object.keys(all);
  const off = keys.filter((k) => !SCALE_SIZES.has(k.split("/")[0]));
  expect(off, `sizes off the scale:\n${off.map((k) => `${k}  ${all[k]}`).join("\n")}`).toEqual([]);
  if (process.env.TYPE_STYLES_LOG) console.log(keys.map((k) => `${k}  ${all[k]}`).join("\n"));
  expect(
    keys.length,
    `type styles in use:\n${keys.map((k) => `${k}  ${all[k]}`).join("\n")}`,
  ).toBeLessThanOrEqual(MAX_STYLES);
});
