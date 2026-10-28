// Console push-vs-poll measurement against a running console process (driven by
// scripts/research/front_push_bench.py, run through gpuq).
//
//   node scripts/push-bench.mjs --base http://127.0.0.1:18991 --model ID --tabs 10 --push 1
//        [--mode events|reconnect] [--idle-s 40] [--cycles 3] [--load-model ID2] [--out FILE]
//
// events: opens N tabs, counts the requests each sends while idle, then drives requests / a model load
//   and records when each tab's live pill (data-testid=live-phase) shows the change (event-to-screen).
// reconnect: one tab; a driver kills and restarts the engine meanwhile; records how long after the engine
//   answers again the pill is back to normal.
import { writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { chromium } from "@playwright/test";

const { values: a } = parseArgs({
  options: {
    base: { type: "string" },
    model: { type: "string" },
    tabs: { type: "string", default: "1" },
    push: { type: "string", default: "1" },
    mode: { type: "string", default: "events" },
    "idle-s": { type: "string", default: "40" },
    cycles: { type: "string", default: "3" },
    "load-model": { type: "string", default: "" },
    tokens: { type: "string", default: "700" },
    seconds: { type: "string", default: "80" },
    out: { type: "string", default: "" },
  },
});
const TABS = Number(a.tabs);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const classify = (text) => {
  const s = text.toLowerCase();
  if (/idle/.test(s)) return "idle";
  if (/load/.test(s)) return "loading";
  if (/decod/.test(s)) return "decode";
  if (/prefill|start|queue/.test(s)) return "arrive";
  if (/offline|restart|connect|switch|paus|unauth/.test(s)) return "away";
  return "other:" + text;
};

const REC = `
(() => {
  window.__ph = [];
  const read = () => {
    const el = document.querySelector('[data-testid="live-phase"]');
    return el ? (el.getAttribute("data-phase") || el.textContent || "") : "";
  };
  let last = null;
  const tick = () => {
    const v = read();
    if (v !== last) { last = v; window.__ph.push({ t: Date.now(), v }); }
  };
  new MutationObserver(tick).observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
  setInterval(tick, 20);
})();`;

async function streamChat(maxTokens, prompt) {
  const t0 = Date.now();
  const res = await fetch(`${a.base}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: a.model,
      stream: true,
      max_tokens: maxTokens,
      ignore_eos: true,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!res.ok) throw new Error(`chat HTTP ${res.status}`);
  let first = null;
  const dec = new TextDecoder();
  for await (const chunk of res.body) {
    const text = dec.decode(chunk);
    if (first === null && /"content"\s*:\s*"[^"]/.test(text))
      first = Date.now();
  }
  return { t0, first, end: Date.now() };
}

const firstAfter = (ph, t, pred) =>
  ph.find((p) => p.t >= t && pred(classify(p.v)));
const stats = (xs) => {
  const v = xs.filter((x) => x != null).sort((x, y) => x - y);
  if (!v.length) return null;
  const q = (p) => v[Math.min(v.length - 1, Math.floor(p * v.length))];
  return { n: v.length, min: v[0], p50: q(0.5), max: v[v.length - 1] };
};

const browser = await chromium.launch();
const ctx = await browser.newContext({
  locale: "en-US",
  viewport: { width: 1280, height: 800 },
});
await ctx.addInitScript(REC);
const pages = [];
const counts = [];
for (let i = 0; i < TABS; i++) {
  const page = await ctx.newPage();
  const c = { total: 0, byPath: {}, window: "warmup" };
  page.on("request", (r) => {
    const u = new URL(r.url());
    if (!u.pathname.startsWith("/v1") && !u.pathname.startsWith("/console/api"))
      return;
    const key = `${c.window}|${r.method()} ${u.pathname}`;
    c.byPath[key] = (c.byPath[key] ?? 0) + 1;
  });
  await page.goto(`${a.base}/console/?push=${a.push}`);
  pages.push(page);
  counts.push(c);
}
for (const p of pages)
  await p.waitForSelector('[data-testid="live-phase"]', { timeout: 60000 });
await sleep(6000); // first loads, backfills, lazy pages
const out = { push: a.push, tabs: TABS, mode: a.mode };
const setWindow = (w) => counts.forEach((c) => (c.window = w));
const summarizeCounts = (w, seconds) => {
  const per = counts.map((c) =>
    Object.entries(c.byPath)
      .filter(([k]) => k.startsWith(w + "|"))
      .reduce((s, [, n]) => s + n, 0),
  );
  const paths = {};
  for (const c of counts)
    for (const [k, n] of Object.entries(c.byPath))
      if (k.startsWith(w + "|"))
        paths[k.slice(w.length + 1)] = (paths[k.slice(w.length + 1)] ?? 0) + n;
  return {
    seconds,
    perTab: per,
    perTabPerMin: per.map((n) => +((n * 60) / seconds).toFixed(1)),
    paths,
  };
};
const pushState = async () =>
  Promise.all(
    pages.map((p) => p.evaluate(() => globalThis.__yunshuPush?.() ?? null)),
  );

if (a.mode === "events") {
  const idleS = Number(a["idle-s"]);
  setWindow("idle");
  out.idleStart = Date.now();
  await sleep(idleS * 1000);
  out.idleEnd = Date.now();
  out.idle = summarizeCounts("idle", idleS);

  setWindow("events");
  out.eventsStart = Date.now();
  const arrive = [],
    first = [],
    idleBack = [];
  const runs = [];
  for (let i = 0; i < Number(a.cycles); i++) {
    const r = await streamChat(
      160,
      `Write a numbered list of 40 short facts about the sea. Run ${i}.`,
    );
    runs.push(r);
    await sleep(4000);
    const phs = await Promise.all(
      pages.map((p) => p.evaluate(() => window.__ph)),
    );
    phs.forEach((ph, k) => {
      const ar = firstAfter(ph, r.t0, (c) => c === "arrive" || c === "decode");
      const de = firstAfter(ph, r.t0, (c) => c === "decode");
      arrive.push(ar ? ar.t - r.t0 : null);
      first.push(de && r.first ? de.t - r.first : null);
      const id = firstAfter(ph, r.end, (c) => c === "idle");
      idleBack.push(id ? id.t - r.end : null);
    });
  }
  out.latencyMs = {
    requestArrival_sent_to_pill: stats(arrive),
    firstToken_clientFirstToken_to_decodePill: stats(first),
    requestEnd_to_idlePill: stats(idleBack),
    clientTtftMs: stats(runs.map((r) => (r.first ? r.first - r.t0 : null))),
    clientRunMs: stats(runs.map((r) => r.end - r.t0)),
    // a tab that never saw the request at all (poll interval longer than the request)
    arrivalMissed: arrive.filter((x) => x == null).length,
    arrivalSamples: arrive.length,
  };
  if (a["load-model"]) {
    // a cell must load for real: drop the model a previous cell left loaded
    await fetch(`${a.base}/v1/models/unload`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: a["load-model"] }),
    }).catch(() => undefined);
    await sleep(2000);
    const t0 = Date.now();
    const res = await fetch(`${a.base}/v1/models/load`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: a["load-model"] }),
    });
    const tr = Date.now();
    await sleep(3000);
    const phs = await Promise.all(
      pages.map((p) => p.evaluate(() => window.__ph)),
    );
    out.modelLoad = {
      http: res.status,
      loadMs: tr - t0,
      loadingPill_after_send: stats(
        phs.map((ph) => {
          const f = firstAfter(ph, t0, (c) => c === "loading");
          return f ? f.t - t0 : null;
        }),
      ),
      idlePill_after_response: stats(
        phs.map((ph) => {
          const f = firstAfter(ph, tr - 1, (c) => c === "idle");
          return f ? f.t - tr : null;
        }),
      ),
    };
  }
  out.eventsEnd = Date.now();
  out.events = summarizeCounts(
    "events",
    (out.eventsEnd - out.eventsStart) / 1000,
  );
  out.pushState = await pushState();
} else {
  // reconnect: the driver restarts the engine while this watches
  setWindow("reconnect");
  const probes = [];
  const end = Date.now() + Number(a.seconds) * 1000;
  console.log("READY");
  while (Date.now() < end) {
    const t = Date.now();
    let ok = false;
    try {
      const r = await fetch(`${a.base}/v1/yunshu/status`, {
        signal: AbortSignal.timeout(1500),
      });
      ok = r.ok;
    } catch {}
    probes.push({ t, ok });
    await sleep(100);
  }
  const ph = await pages[0].evaluate(() => window.__ph);
  // first ok probe after the first failure that is followed by a run of ok probes
  const firstFail = probes.findIndex((p) => !p.ok);
  const up = firstFail < 0 ? null : probes.slice(firstFail).find((p) => p.ok);
  const lastAway = [...ph].reverse().find((p) => classify(p.v) === "away");
  const backIdle = up
    ? ph.find((p) => p.t >= up.t && classify(p.v) === "idle")
    : null;
  out.reconnect = {
    engineDownAt: firstFail < 0 ? null : probes[firstFail].t,
    engineUpAt: up?.t ?? null,
    downMs: up ? up.t - probes[firstFail].t : null,
    pillAway: ph.filter((p) => classify(p.v) === "away").length > 0,
    pillIdleAfterUp_ms: backIdle && up ? backIdle.t - up.t : null,
    pillTimeline: ph.map((p) => ({ dt: p.t - (probes[0]?.t ?? 0), v: p.v })),
  };
  out.reconnectCounts = summarizeCounts("reconnect", Number(a.seconds));
  out.pushState = await pushState();
}
await browser.close();
const json = JSON.stringify(out, null, 1);
if (a.out) writeFileSync(a.out, json);
console.log(json);
