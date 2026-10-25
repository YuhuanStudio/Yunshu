// Debug-section screenshots every 2 s against a live observing engine, plus a "never ahead" check of the
// Overview's displayed token count against /debug/requests read just before and just after it.
// Usage: node scripts/uihonesty-shots.mjs CONSOLE_URL ENGINE_URL OUT_DIR SECONDS
import { chromium } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";

const [base, engine, out, secondsArg] = process.argv.slice(2);
const seconds = Number(secondsArg ?? 90);
mkdirSync(out, { recursive: true });

export async function inFlight() {
  try {
    const r = await fetch(`${engine}/debug/requests`);
    return (await r.json()).in_flight ?? [];
  } catch {
    return [];
  }
}
const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
});
const req = await ctx.newPage();
const over = await ctx.newPage();
const errors = [];
for (const p of [req, over]) p.on("pageerror", (e) => errors.push(e.message));
await req.goto(`${base}/console/#/requests`);
await over.goto(`${base}/console/#/overview`);
await req.getByTestId("observe-panel").waitFor({ timeout: 60000 });

const samples = [];
const t0 = Date.now();
let i = 0;
while ((Date.now() - t0) / 1000 < seconds) {
  const rows = await req.getByTestId("observe-row").evaluateAll((els) =>
    els.map((el) => ({
      id: el.firstElementChild?.textContent ?? "",
      differs: el.getAttribute("data-differs") === "true",
      cells: [...el.querySelectorAll("td[data-state]")].map((td) => ({
        state: td.getAttribute("data-state"),
        text: td.textContent,
      })),
    })),
  );
  await req
    .getByTestId("observe-panel")
    .screenshot({ path: `${out}/debug-${String(i).padStart(3, "0")}.png` });
  const before = await inFlight();
  const shown = await over.evaluate(() =>
    [...document.querySelectorAll("[data-live-list] > li")].map((li) => {
      const text = (li.textContent ?? "").replace(/,/g, "");
      const m = /(?:^|[^\w])(\d+)\s*tok(?!\/)/.exec(
        text.replace(/req_[0-9a-f]+/g, ""),
      );
      return { text: text.slice(0, 60), count: m ? Number(m[1]) : null };
    }),
  );
  const after = await inFlight();
  const ahead = [];
  for (const row of shown) {
    for (const b of before) {
      const a = after.find((x) => x.request_id === b.request_id);
      if (
        !a ||
        (!row.text.includes(b.request_id.slice(-10)) &&
          !row.text.includes(b.request_id))
      )
        continue;
      if (b.phase !== "decode" || a.phase !== "decode" || row.count == null)
        continue;
      // The page shows a count the engine reported earlier; it must never exceed what the engine had after.
      if (row.count > a.generated)
        ahead.push({
          id: b.request_id,
          shown: row.count,
          engine_after: a.generated,
          engine_before: b.generated,
        });
    }
  }
  samples.push({ i, t: (Date.now() - t0) / 1000, rows, shown, ahead });
  i++;
  await new Promise((r) => setTimeout(r, 2000));
}
await browser.close();
const mismatches = samples.flatMap((s) =>
  s.rows.filter((r) => r.differs).map((r) => ({ i: s.i, ...r })),
);
const aheadAll = samples.flatMap((s) => s.ahead.map((a) => ({ i: s.i, ...a })));
const twoRows = samples.filter((s) => s.rows.length >= 2).length;
const result = {
  samples: samples.length,
  samples_with_two_rows: twoRows,
  red_mismatches: mismatches.length,
  ahead_of_engine: aheadAll.length,
  mismatches,
  aheadAll,
  errors,
};
writeFileSync(`${out}/result.json`, JSON.stringify(result, null, 2));
writeFileSync(`${out}/samples.json`, JSON.stringify(samples, null, 2));
console.log(
  JSON.stringify({
    ...result,
    mismatches: mismatches.slice(0, 5),
    aheadAll: aheadAll.slice(0, 5),
  }),
);
process.exit(
  mismatches.length || aheadAll.length || twoRows < 5 || errors.length ? 1 : 0,
);
