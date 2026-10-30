import { expect, test, type BrowserContext, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Hover / focus affordance audit (user, 2026-10-29: 「一堆地方的鼠標浮動效果也沒做好」).
 *
 * Every interactive element (button, link, tab, menu item, switch, summary, clickable row or card, ...)
 * must show a visible change on hover, a pointer cursor (a text cursor on text fields) and a visible
 * focus ring on keyboard focus. Every NON-interactive element (plain cards, stat tiles, static rows,
 * headings, charts) must show no hover affordance at all: no border / shadow / background / transform /
 * colour change and no pointer cursor. The status band is wrapped by one overlay button, so the band is
 * one interactive thing and the pills inside it are part of it.
 *
 * Elements are sampled per distinct signature (tag, role, classes) so a table with 200 identical rows
 * costs two hovers, not 200. Pages run in parallel tabs of one context.
 *
 * HOVER_REPORT=/path/file.txt writes the full violation list; HOVER_SHOTS=/dir HOVER_SHOT_TAG=before|after
 * also writes a few hover screenshots.
 */

test.use({ viewport: { width: 1440, height: 900 } });

// HOVER_EXTRA=1 also audits the Playground and the docs (report only; not part of the default gate).
const EXTRA = process.env.HOVER_EXTRA === "1";
const PAGES = [
  ...(EXTRA ? ["playground", "docs/getting-started/install"] : []),
  "overview",
  "requests",
  "logs",
  "diagnostics",
  "models",
  "downloads",
  "cache",
  "keys",
  "settings",
  "api",
] as const;

const GB = 2 ** 30;
const T = 1_800_000_000;

const json = (body: unknown, status = 200) => ({
  status,
  contentType: "application/json",
  body: JSON.stringify(body),
});

const models = [
  {
    id: "org/qwen-mlx",
    type: "text",
    loaded: true,
    loading: false,
    pinned: true,
    size_gb: 8,
    expires_in_s: null,
  },
  {
    id: "org/vision-vlm",
    type: "vision-vlm",
    loaded: true,
    loading: false,
    pinned: false,
    size_gb: 10,
    expires_in_s: 900,
  },
  {
    id: "org/idle-mlx",
    type: "text",
    loaded: false,
    loading: false,
    pinned: false,
    size_gb: 4,
    expires_in_s: null,
  },
];

const items = [
  {
    request_id: "qa-queue-01",
    elapsed_s: 6.2,
    phase: "queued",
    model: "org/qwen-mlx",
    queue_position: 2,
    queue_est_wait_ms: 1800,
  },
  {
    request_id: "qa-prefill-01",
    elapsed_s: 3.4,
    phase: "prefill",
    model: "org/qwen-mlx",
    prompt_tokens: 4000,
    cached_tokens: 1000,
    processed_tokens: 2500,
    percent: 62.5,
    eta_s: 1.2,
    tokens_per_second: 900,
  },
  {
    request_id: "qa-decode-01",
    elapsed_s: 9,
    phase: "decode",
    model: "org/qwen-mlx",
    prompt_tokens: 600,
    cached_tokens: 200,
    completion_tokens: 42,
    tokens_per_second: 28.5,
  },
];

const ring = (i: number) => ({
  request_id: `ring-req-0123456789abcdef-${String(i).padStart(3, "0")}`,
  t: T - i * 7,
  path: "/v1/chat/completions",
  model: "org/qwen-mlx",
  status: i % 5 === 0 ? 500 : 200,
  finish_reason: "stop",
  stream: true,
  t0_wall: T - 10 - i * 7,
  offsets_ms: {
    arrive: 0,
    admit: 40,
    first_token: 340,
    last_token: 2340,
    done: 2360,
  },
  queue_wait_ms: 40,
  ttft_ms: 300 + i * 11,
  prompt_tokens: 4000,
  cached_tokens: 1000,
  completion_tokens: 80,
  prefill_tps: 800,
  decode_tps: 40,
  cache: { tier: "ram", reload_ms: null },
  speculative: { mode: "mtp", acceptance_rate: 0.8, rounds: 20 },
  cancelled: false,
});

const cfgRow = (o: Record<string, unknown>) => ({
  value: null,
  default: null,
  source: "default",
  stability: "stable",
  category: "server",
  description: "A setting.",
  type: "str",
  choices: [],
  applies: "live",
  minimum: null,
  secret: false,
  ...o,
});

const logRec = (id: number, level: string, msg: string) => ({
  id,
  t: T + id,
  level,
  logger: id % 2 ? "yunshu.engine" : "uvicorn.access",
  msg,
});
const logs = [
  logRec(1, "INFO", "model loaded org/qwen-mlx"),
  logRec(2, "WARNING", "memory pressure 0.91"),
  logRec(3, "ERROR", "request failed"),
  ...Array.from({ length: 30 }, (_, i) => logRec(4 + i, "INFO", `step ${i}`)),
];

const dl = (id: string, state: string, over: Record<string, unknown> = {}) => ({
  id,
  repo: `org/${id}`,
  revision: null,
  allow_patterns: null,
  state,
  error: state === "failed" ? "ConnectionError: reset" : null,
  path: `/models/org/${id}`,
  bytes_total: 8 * GB,
  bytes_done: 3 * GB,
  files_total: 12,
  files_done: 3,
  active_files: [],
  rate_bps: state === "running" ? 45 * 2 ** 20 : null,
  eta_s: state === "running" ? 108 : null,
  created: 1,
  started: 1,
  finished: null,
  registered: false,
  already_present: false,
  ...over,
});

async function install(context: BrowserContext) {
  await context.addInitScript(() => {
    localStorage.setItem("yunshu.console.url", location.origin);
  });
  await context.route("**/openapi.json", (r) => r.fulfill(json({ paths: {} })));
  await context.route("**/v1/**", async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    const method = req.method();
    const nf = () => route.fulfill(json({ detail: "Not Found" }, 404));
    if (method !== "GET") return nf();
    switch (path) {
      case "/v1/yunshu/status":
        return route.fulfill(
          json({
            object: "yunshu.status",
            version: "fixture-1.0",
            state: "ready",
            uptime_s: 93_784,
            load_error: null,
            models,
            memory: {
              active_gb: 25,
              cache_gb: 2,
              peak_gb: 30,
              total_gb: 64,
              pressure: 0.3,
            },
            requests: { active: 3, queued: 1, prefill: 1, decode: 1, items },
            last: {
              request_id: "qa-done-1",
              prompt_tokens: 1000,
              completion_tokens: 201,
              cached_tokens: 250,
              prefill_tps: 800,
              decode_tps: 41,
              ttft_ms: 310,
              t: T,
              speculative: { mode: "mtp", acceptance_rate: 0.82, rounds: 12 },
            },
            throughput: {
              window_s: 60,
              requests: 4,
              prompt_tokens: 4000,
              completion_tokens: 800,
              live_decode_tps: 34,
              mean_prefill_tps: 800,
              mean_decode_tps: 35,
            },
          }),
        );
      case "/v1/yunshu/requests/recent":
        return route.fulfill(
          json({
            object: "list",
            data: Array.from({ length: 14 }, (_, i) => ring(i + 1)),
            count: 14,
            capacity: 512,
          }),
        );
      case "/v1/yunshu/config":
        return route.fulfill(
          json({
            object: "yunshu.config",
            settings: [
              cfgRow({
                name: "YUNSHU_LOG_LEVEL",
                value: "INFO",
                default: "INFO",
                type: "enum",
                choices: ["DEBUG", "INFO", "WARNING"],
                description: "Log level.",
              }),
              cfgRow({
                name: "YUNSHU_QUEUE_LIMIT",
                value: 64,
                default: 64,
                type: "int",
                minimum: 0,
                description: "Requests in flight at once.",
              }),
              cfgRow({
                name: "YUNSHU_PREFIX_MAX_ENTRIES",
                value: 64,
                default: 64,
                type: "int",
                minimum: 1,
                applies: "reload",
              }),
              cfgRow({
                name: "YUNSHU_PREFILL_GDN",
                value: false,
                default: false,
                type: "bool",
                stability: "experimental",
              }),
            ],
            warnings: [],
            experimental_count: 1,
            experimental_max: 8,
          }),
        );
      case "/v1/yunshu/service":
        return route.fulfill(
          json({
            label: "ai.yunshu.server",
            plist: "/Users/me/Library/LaunchAgents/ai.yunshu.server.plist",
            installed: true,
            loaded: true,
            pid: 4242,
            state: "running",
            last_exit_code: 0,
            log: "/Users/me/.yunshu/logs/service.log",
            under_launchd: true,
            version: "0.1.4",
            uptime_s: 93784,
            cli: {
              status: "yunshu service status",
              restart: "yunshu service restart",
              install: "yunshu service install",
            },
          }),
        );
      case "/v1/yunshu/cors":
        return route.fulfill(
          json({
            origins: ["http://localhost:3000", "http://localhost:8000"],
            wildcard: false,
            credentials: true,
            source: "default",
            default: "http://localhost:3000,http://localhost:8000",
            warnings: [],
          }),
        );
      case "/v1/yunshu/keys":
        return route.fulfill(
          json({
            object: "list",
            data: [
              {
                id: "key_aaa",
                name: "laptop",
                prefix: "ysk-ab12",
                created: 1790000000,
                enabled: true,
                scopes: ["infer"],
                expires: null,
                expired: false,
                quotas: {
                  requests_per_day: 1000,
                  tokens_per_day: null,
                  max_concurrent: 2,
                },
                last_used: T - 120,
                window: { requests: 640, tokens: 120000, inflight: 0 },
              },
              {
                id: "key_bbb",
                name: "CI",
                prefix: "ysk-cd34",
                created: 1790000000,
                enabled: false,
                scopes: ["infer", "admin"],
                expires: T + 86400 * 20,
                expired: false,
                quotas: {
                  requests_per_day: null,
                  tokens_per_day: 500000,
                  max_concurrent: null,
                },
                last_used: null,
                window: { requests: 0, tokens: 0, inflight: 0 },
              },
            ],
          }),
        );
      case "/v1/yunshu/usage":
        return route.fulfill(
          json({
            object: "list",
            data: Array.from({ length: 10 }, (_, i) => ({
              key: "key_aaa",
              name: "laptop",
              day: new Date(Date.now() - i * 86400000)
                .toISOString()
                .slice(0, 10),
              requests: 100 + i * 17,
              prompt_tokens: 40000 + i * 900,
              completion_tokens: 9000,
              cached_tokens: 0,
              errors: 0,
            })),
          }),
        );
      case "/v1/yunshu/logs":
        return route.fulfill(
          json({
            records: logs,
            next_id: logs.length,
            dropped: 0,
            capacity: 2000,
            server_time: T + 70,
          }),
        );
      case "/v1/yunshu/logs/stream":
        return route.fulfill({
          status: 200,
          contentType: "text/event-stream",
          body: ": keepalive\n\n",
        });
      case "/v1/yunshu/downloads":
        return route.fulfill(
          json({
            downloads: [
              dl("running-one", "running"),
              dl("broken", "failed"),
              dl("finished", "done", {
                bytes_done: 8 * GB,
                files_done: 12,
                registered: true,
              }),
            ],
            active: 1,
            free_bytes: 212 * GB,
            models_dir: "/Users/me/.yunshu/models",
          }),
        );
      case "/v1/yunshu/models/local":
        return route.fulfill(
          json({
            models: [
              {
                id: "org/on-disk",
                path: "/models/org/on-disk",
                source: "models_dir",
                size_bytes: 4.4 * GB,
                model_type: "qwen3",
                kind: "llm",
                architecture: "Qwen3",
                parameters: "8B",
                quantization: { bits: 4, group_size: 64 },
                context_length: 32768,
                capabilities: ["tools", "reasoning"],
                complete: true,
                complete_reason: null,
                registered_as: null,
                loaded: false,
              },
              {
                id: "org/qwen-mlx",
                path: "/models/org/qwen-mlx",
                source: "models_dir",
                size_bytes: 8 * GB,
                complete: true,
                registered_as: "org/qwen-mlx",
                loaded: true,
              },
            ],
            total_bytes: 12 * GB,
            models_dir: "/models",
            free_bytes: 212 * GB,
          }),
        );
      case "/v1/yunshu/cache/tiers":
        return route.fulfill(
          json({
            enabled: true,
            caches: [
              {
                model: "org/qwen-mlx",
                tiers: [
                  {
                    name: "ram",
                    used_bytes: 2.4 * GB,
                    cap_bytes: 8 * GB,
                    entries: 14,
                    hits: 80,
                  },
                  {
                    name: "ssd",
                    used_bytes: 12 * GB,
                    cap_bytes: 64 * GB,
                    entries: 31,
                    hits: 9,
                  },
                ],
                lookups: { hit: 95, miss: 25, by_tier: { ram: 80, ssd: 15 } },
                entries: Array.from({ length: 12 }, (_, i) => ({
                  key: (0xa1b2c3d4 + i * 4097).toString(16).padStart(8, "0"),
                  tokens: 1000 + i * 700,
                  bytes: (200 + i * 30) * 1e6,
                  tier: i % 4 === 0 ? "warm" : "ram",
                  lru_rank: i,
                  hits: i % 5,
                  last_hit_age_s: i % 3 === 0 ? null : i * 41,
                })),
                entries_truncated: false,
              },
            ],
          }),
        );
      default:
        return nf();
    }
  });
  await context.route("**/debug/**", (r) =>
    r.fulfill(json({ detail: "off" }, 404)),
  );
}

type Found = {
  kind: "interactive" | "disabled" | "static";
  id: number;
  label: string;
  how: string;
  hasFocusRing?: boolean;
};

/** Installs window.__hv with the in-page helpers (runs in the page, so no Node closures). */
async function installHelpers(page: Page) {
  await page.evaluate(() => {
    const PROPS = [
      "backgroundColor",
      "borderTopColor",
      "borderRightColor",
      "borderBottomColor",
      "borderLeftColor",
      "color",
      "boxShadow",
      "transform",
      "opacity",
      "textDecorationLine",
      "textDecorationColor",
      "textDecorationThickness",
      "filter",
      "backgroundImage",
      "outlineStyle",
    ] as const;
    const CHILD_PROPS = [
      "backgroundColor",
      "color",
      "opacity",
      "transform",
      "boxShadow",
      "textDecorationLine",
    ] as const;
    const INTERACTIVE_SEL = [
      "button",
      "a[href]",
      "summary",
      "select",
      "textarea",
      "input:not([type=hidden])",
      "[role=button]",
      "[role=link]",
      "[role=tab]",
      "[role=menuitem]",
      "[role=menuitemcheckbox]",
      "[role=menuitemradio]",
      "[role=switch]",
      "[role=checkbox]",
      "[role=radio]",
      "[role=combobox]",
      "[role=option]",
      "[role=treeitem]",
    ].join(",");

    const els: Element[] = [];
    const idOf = (el: Element) => {
      let i = els.indexOf(el);
      if (i < 0) {
        els.push(el);
        i = els.length - 1;
      }
      return i;
    };
    const visible = (el: Element) => {
      const r = el.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) return false;
      if (el.closest("[aria-hidden=true],[hidden],[inert]")) return false;
      for (let e: Element | null = el; e; e = e.parentElement) {
        const cs = getComputedStyle(e);
        if (cs.display === "none" || cs.visibility === "hidden") return false;
      }
      return true;
    };
    const reactProps = (el: Element) => {
      const k = Object.keys(el).find((x) => x.startsWith("__reactProps$"));
      return k
        ? ((el as unknown as Record<string, Record<string, unknown>>)[k] ?? {})
        : {};
    };
    // A block with an onClick handler is a clickable row/card. Tooltip triggers and inline wrappers
    // carry handlers too (Radix) but are not clickable things.
    const clickable = (el: Element) =>
      typeof reactProps(el).onClick === "function" &&
      /^(TR|LI|DIV|SECTION|ARTICLE|TD)$/.test(el.tagName) &&
      !/^(closed|delayed-open|instant-open)$/.test(
        el.getAttribute("data-state") ?? "",
      );
    const isDisabled = (el: Element) =>
      (el as HTMLButtonElement).disabled === true ||
      el.getAttribute("aria-disabled") === "true" ||
      el.hasAttribute("data-disabled") ||
      !!el.closest("fieldset[disabled]");
    const label = (el: Element) => {
      const tag = el.tagName.toLowerCase();
      const role = el.getAttribute("role");
      const aria = el.getAttribute("aria-label");
      const tid = el.getAttribute("data-testid");
      const text = (el.textContent ?? "").replace(/\s+/g, " ").trim();
      const cls = (typeof el.className === "string" ? el.className : "")
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 3)
        .join(".");
      let s = tag;
      if (role) s += `[role=${role}]`;
      if (tid) s += `[testid=${tid}]`;
      else if (aria) s += `[aria-label="${aria.slice(0, 30)}"]`;
      else if (text) s += `["${text.slice(0, 28)}"]`;
      if (cls) s += `.${cls}`;
      const parent = el.parentElement;
      if (parent && !tid && !aria && !text) {
        const pt = parent.getAttribute("data-testid");
        if (pt) s += ` in[testid=${pt}]`;
      }
      return s;
    };
    const sig = (el: Element) =>
      `${el.tagName}|${el.getAttribute("role")}|${typeof el.className === "string" ? el.className : ""}|${el.closest("footer,nav,aside,main,header")?.tagName}`;

    const snap = (el: Element) => {
      const out: Record<string, string> = {};
      const cs = getComputedStyle(el);
      for (const p of PROPS) out[`self.${p}`] = String(cs[p]);
      // A border colour only shows where there is a border.
      for (const side of ["Top", "Right", "Bottom", "Left"] as const) {
        const width = parseFloat(
          cs[`border${side}Width` as "borderTopWidth"] || "0",
        );
        if (!width) out[`self.border${side}Color`] = "none";
      }
      for (const pseudo of ["::before", "::after"] as const) {
        const pc = getComputedStyle(el, pseudo);
        if (pc.content && pc.content !== "none" && pc.content !== "normal") {
          out[`${pseudo}.backgroundColor`] = pc.backgroundColor;
          out[`${pseudo}.opacity`] = pc.opacity;
          out[`${pseudo}.boxShadow`] = pc.boxShadow;
        }
      }
      // A composite field draws its border on the wrapper around the input.
      if (el instanceof HTMLInputElement && el.parentElement) {
        const pc = getComputedStyle(el.parentElement);
        out["wrap.borderColor"] = parseFloat(pc.borderTopWidth)
          ? pc.borderTopColor
          : "none";
        out["wrap.boxShadow"] = pc.boxShadow;
        out["wrap.backgroundColor"] = pc.backgroundColor;
      }
      const kids = el.querySelectorAll("*");
      for (let i = 0; i < kids.length && i < 40; i++) {
        // A row that reveals its own action on hover (data-row-actions) is hover-by-design.
        // A chart reads out the point under the pointer: data inspection, not a hover state.
        if (
          kids[i].closest(
            "[data-row-actions], [data-yunui=sparkline-tip], [data-yunui=column-chart], .yunui-metric, svg",
          )
        )
          continue;
        const kc = getComputedStyle(kids[i]);
        for (const p of CHILD_PROPS) out[`d${i}.${p}`] = String(kc[p]);
        if (kids[i] instanceof SVGElement) {
          out[`d${i}.stroke`] = kc.stroke;
          out[`d${i}.fill`] = kc.fill;
        }
      }
      return out;
    };
    // A shadow list made only of zero-size transparent entries (Tailwind's empty ring) is no shadow.
    const shadow = (v: string) =>
      v
        .split(/,(?![^(]*\))/)
        .map((p) => p.trim())
        .filter(
          (p) =>
            p !== "none" &&
            !/^rgba\(0, 0, 0, 0\) 0px 0px 0px 0px( inset)?$/.test(p) &&
            !/^0px 0px 0px 0px rgba\(0, 0, 0, 0\)( inset)?$/.test(p),
        )
        .join(",") || "none";
    const ringProps = (el: Element) => {
      const cs = getComputedStyle(el);
      return [
        cs.outlineStyle,
        cs.outlineWidth,
        shadow(cs.boxShadow),
        cs.borderTopColor,
        cs.borderBottomColor,
        cs.backgroundColor,
      ].join("|");
    };

    type Item = {
      id: number;
      kind: string;
      label: string;
      how: string;
    };
    const rest = new Map<number, Record<string, string>>();
    const noisy = new Map<number, Set<string>>();
    const ringRest = new Map<number, string[]>();

    const w = window as unknown as Record<string, unknown>;
    w.__hv = {
      /** Enumerate sampled interactive / disabled / static elements and take two rest snapshots. */
      collect(rootSel?: string): Item[] {
        const all = rootSel
          ? [...document.querySelectorAll(rootSel)].flatMap((r) => [
              r,
              ...r.querySelectorAll("*"),
            ])
          : [...document.querySelectorAll("*")];
        // The status islands: each capsule's own overlay button is the interactive thing; the pills under them are not.
        const band = document.querySelector("[data-testid=status-band]");
        const inFooter = (el: Element) => !!band?.contains(el);
        const inter: Item[] = [];
        const dis: Item[] = [];
        const stat: Item[] = [];
        const seen = new Map<string, number>();
        const take = (list: Item[], el: Element, how: string, cap: number) => {
          const key = `${list === inter ? "i" : list === dis ? "d" : "s"}|${how}|${sig(el)}`;
          const n = seen.get(key) ?? 0;
          if (n >= cap) return;
          seen.set(key, n + 1);
          list.push({
            id: idOf(el),
            kind:
              list === inter
                ? "interactive"
                : list === dis
                  ? "disabled"
                  : "static",
            label: label(el),
            how,
          });
        };
        const interactiveRoots = new Set<Element>();
        for (const el of all) {
          if (el.tagName === "svg" || el instanceof SVGElement) continue;
          if (!visible(el)) continue;
          let how = "";
          if (el.matches(INTERACTIVE_SEL)) {
            if (el instanceof HTMLInputElement && /hidden/.test(el.type))
              continue;
            how = "native/role";
          } else if (clickable(el)) {
            how = "onClick";
          } else if (
            getComputedStyle(el).cursor === "pointer" &&
            (!el.parentElement ||
              getComputedStyle(el.parentElement).cursor !== "pointer")
          ) {
            how = "cursor:pointer";
          } else if (el.matches(".card-interactive")) {
            how = "card-interactive";
          }
          if (how) interactiveRoots.add(el);
        }
        const insideInteractive = (el: Element) => {
          for (let e: Element | null = el; e; e = e.parentElement)
            if (interactiveRoots.has(e)) return true;
          return false;
        };
        for (const el of interactiveRoots) {
          if (inFooter(el) && !el.matches("[data-testid^=footer-trigger]"))
            continue;
          const how = el.matches(INTERACTIVE_SEL)
            ? "native/role"
            : clickable(el)
              ? "onClick"
              : "cursor:pointer";
          // The sidebar/header chrome repeats on every page: sampled once per signature anyway.
          take(isDisabled(el) ? dis : inter, el, how, 2);
        }
        const STATIC_SEL = [
          ".card",
          ".glass-card",
          ".stat-card",
          "[data-stat-grid] > *",
          "[data-yunui=card]",
          "[data-slot=card]",
          "tbody tr",
          "thead th",
          "thead tr",
          "[role=row]",
          "h1",
          "h2",
          "h3",
          "h4",
          "li",
          "[role=status]",
          // (Charts are left out: a time-series readout under the pointer is data inspection, not a hover state.)
          "[class*=badge]",
          "[class*=pill]",
          "[class*=chip]",
          "section",
          "figure",
          "dt",
          "dd",
          "p",
        ].join(",");
        for (const el of all.filter((e) => e.matches(STATIC_SEL))) {
          if (!visible(el)) continue;
          if (insideInteractive(el)) continue;
          if (inFooter(el)) continue;
          // A row that contains an interactive control is still a static row.
          take(stat, el, "static", 1);
        }
        const list = [...inter, ...dis, ...stat];
        // Two rest snapshots 250 ms apart: whatever animates on its own is not a hover effect.
        for (const it of list) {
          const el = els[it.id];
          rest.set(it.id, snap(el));
        }
        return list;
      },
      collectNoise(ids: number[]) {
        for (const id of ids) {
          const b = snap(els[id]);
          const a = rest.get(id) ?? {};
          const s = new Set<string>();
          for (const k of new Set([...Object.keys(a), ...Object.keys(b)]))
            if (a[k] !== b[k]) s.add(k);
          noisy.set(id, s);
        }
      },
      /** Scroll the element into view and pick a point that lands on it (or one of its descendants). */
      point(id: number, wantSelf: boolean) {
        const el = els[id] as HTMLElement;
        el.scrollIntoView({ block: "center", inline: "nearest" });
        const r = el.getBoundingClientRect();
        const cands: [number, number][] = [
          [
            r.left + Math.min(8, r.width / 2),
            r.top + Math.min(8, r.height / 2),
          ],
          [r.left + r.width / 2, r.top + r.height / 2],
          [
            r.right - Math.min(8, r.width / 2),
            r.bottom - Math.min(8, r.height / 2),
          ],
          [r.left + r.width / 4, r.top + r.height / 2],
          [r.left + (3 * r.width) / 4, r.top + r.height / 2],
        ];
        let fallback: [number, number] | null = null;
        for (const [x, y] of cands) {
          if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) continue;
          const hit = document.elementFromPoint(x, y);
          if (!hit) continue;
          if (hit === el) return { x, y, ok: true };
          if (wantSelf) {
            const inner = hit.closest(INTERACTIVE_SEL);
            if (inner && inner !== el && el.contains(inner)) continue;
          }
          if (el.contains(hit) && !fallback) fallback = [x, y];
        }
        if (fallback && !wantSelf)
          return { x: fallback[0], y: fallback[1], ok: true };
        if (fallback) return { x: fallback[0], y: fallback[1], ok: true };
        return { x: 0, y: 0, ok: false };
      },
      /** Properties that differ from rest (minus self-animating ones) plus the cursor under the pointer. */
      diff(id: number, x: number, y: number) {
        const el = els[id];
        void getComputedStyle(el).color; // a style recalc starts the transitions
        // Tabs open in parallel and a background tab does not tick transitions: jump to their end.
        const running = document
          .getAnimations()
          .filter((a) => a instanceof CSSTransition);
        const ms = (v: string) =>
          Math.max(
            ...v
              .split(",")
              .map((t) =>
                t.includes("ms") ? parseFloat(t) : parseFloat(t) * 1000,
              ),
          );
        const durMs = ms(getComputedStyle(el).transitionDuration);
        for (const a of running) a.finish();
        const now = snap(el);
        const base = rest.get(id) ?? {};
        const skip = noisy.get(id) ?? new Set<string>();
        const changed: string[] = [];
        const keys: string[] = [];
        for (const k of Object.keys(now))
          if (!skip.has(k) && now[k] !== base[k]) {
            keys.push(k);
            changed.push(
              `${k}: ${String(base[k]).slice(0, 60)} -> ${now[k].slice(0, 60)}`,
            );
          }
        const hit = document.elementFromPoint(x, y);
        const cs = getComputedStyle(el);
        return {
          changed,
          keys,
          cursor: hit ? getComputedStyle(hit).cursor : "?",
          hovered: el.matches(":hover"),
          selected:
            el.getAttribute("aria-selected") === "true" ||
            el.getAttribute("aria-pressed") === "true" ||
            (el.getAttribute("role") === "radio" &&
              el.getAttribute("aria-checked") === "true") ||
            el.getAttribute("aria-current") === "page" ||
            el.getAttribute("data-state") === "active",
          durMs,
        };
      },
      /** Of the keys that differed under the pointer, those that still differ once it has left (live churn). */
      recheck(id: number, keys: string[]) {
        const el = els[id];
        void getComputedStyle(el).color;
        for (const a of document.getAnimations())
          if (a instanceof CSSTransition) a.finish();
        const now = snap(el);
        const base = rest.get(id) ?? {};
        return keys.filter((k) => now[k] !== base[k]);
      },
      /** Tag every tabbable element and remember how it looks at rest. */
      tagFocusables() {
        const sel = INTERACTIVE_SEL + ",[tabindex]";
        for (const el of document.querySelectorAll(sel)) {
          if (!visible(el)) continue;
          const id = idOf(el);
          const chain: string[] = [];
          let e: Element | null = el;
          for (let i = 0; i < 4 && e; i++, e = e.parentElement)
            chain.push(ringProps(e));
          ringRest.set(id, chain);
        }
      },
      focusState() {
        const el = document.activeElement;
        if (!el || el === document.body || el === document.documentElement)
          return null;
        void getComputedStyle(el).color;
        for (const a of document.getAnimations())
          if (a instanceof CSSTransition) a.finish();
        const id = idOf(el);
        const chain = ringRest.get(id);
        const fv = el.matches(":focus-visible");
        const cs = getComputedStyle(el);
        const outline =
          cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) > 0;
        let ringBy = outline ? "outline" : "";
        if (!ringBy && chain) {
          let e: Element | null = el;
          for (let i = 0; i < 4 && e && !ringBy; i++, e = e.parentElement) {
            const c2 = getComputedStyle(e);
            if (
              c2.outlineStyle !== "none" &&
              parseFloat(c2.outlineWidth) > 0 &&
              e !== el
            )
              ringBy = "ancestor-outline";
            else if (ringProps(e) !== chain[i])
              ringBy = i ? "ancestor" : "self-change";
          }
        }
        const restShadow = chain?.[0]?.split("|")[2];
        const ownShadow =
          !!chain &&
          shadow(cs.boxShadow) !== "none" &&
          shadow(cs.boxShadow) !== restShadow;
        return {
          id,
          label: label(el),
          doubleRing: outline && ownShadow,
          focusVisible: fv,
          ringBy,
          known: !!chain,
          disabled: isDisabled(el),
        };
      },
    };
  });
}

/** Elements no hover point could reach (covered, clipped or off screen): listed in the report, not failed. */
const skipped = new Set<string>();

type Violation = {
  cat: string;
  label: string;
  detail: string;
  pages: string[];
};

type Overlay = {
  name: string;
  route: string;
  /** Opens the popover/menu/dialog to audit; only what lives inside it is checked. */
  open: (page: Page) => Promise<void>;
  /** What to audit once open (default: everything the open action added to the page). */
  root?: string;
};
// Whatever the open action adds under an element that was already there.
const OVERLAY_ROOT = "[data-hv-pre] > :not([data-hv-pre])";

async function auditPage(
  context: BrowserContext,
  route: string,
  out: Map<string, Violation>,
  stats: Record<string, number>,
  overlay?: Overlay,
) {
  const where = overlay ? overlay.name : route;
  const page = await context.newPage();
  page.on("pageerror", () => {});
  const add = (cat: string, label: string, detail: string) => {
    const k = `${cat}|${label}`;
    const v = out.get(k);
    if (v) {
      if (!v.pages.includes(where)) v.pages.push(where);
    } else out.set(k, { cat, label, detail, pages: [where] });
  };
  await page.goto(`/console/#/${route}`, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("main", { timeout: 15_000 });
  await page.waitForTimeout(1800);
  if (overlay) {
    await page.evaluate(() => {
      for (const e of document.querySelectorAll("*"))
        e.setAttribute("data-hv-pre", "");
    });
    await page.bringToFront();
    await overlay.open(page);
    await page.waitForTimeout(700);
  }
  await installHelpers(page);
  const list = (await page.evaluate(
    (root) =>
      (
        window as unknown as { __hv: { collect(r?: string): Found[] } }
      ).__hv.collect(root),
    overlay ? (overlay.root ?? OVERLAY_ROOT) : undefined,
  )) as Found[];
  await page.waitForTimeout(260);
  await page.evaluate(
    (ids) =>
      (
        window as unknown as { __hv: { collectNoise(i: number[]): void } }
      ).__hv.collectNoise(ids),
    list.map((l) => l.id),
  );
  stats[`${where}.interactive`] = list.filter(
    (l) => l.kind === "interactive",
  ).length;
  stats[`${where}.disabled`] = list.filter((l) => l.kind === "disabled").length;
  stats[`${where}.static`] = list.filter((l) => l.kind === "static").length;

  for (const it of list) {
    const p = (await page.evaluate(
      ([id, isStatic]) =>
        (
          window as unknown as {
            __hv: { point(i: number, s: boolean): unknown };
          }
        ).__hv.point(id as number, isStatic as boolean),
      [it.id, it.kind === "static"],
    )) as { x: number; y: number; ok: boolean };
    if (!p.ok) {
      stats.covered = (stats.covered ?? 0) + 1;
      skipped.add(`${it.kind} ${it.label} (${where})`);
      continue;
    }
    await page.mouse.move(0, 0);
    await page.mouse.move(p.x, p.y);
    await page.waitForTimeout(70);
    const d = (await page.evaluate(
      ([id, x, y]) =>
        (
          window as unknown as {
            __hv: {
              diff(
                i: number,
                x: number,
                y: number,
              ): {
                changed: string[];
                keys: string[];
                cursor: string;
                hovered: boolean;
                durMs: number;
                property: string;
              };
            };
          }
        ).__hv.diff(id, x, y),
      [it.id, p.x, p.y],
    )) as {
      changed: string[];
      keys: string[];
      cursor: string;
      hovered: boolean;
      durMs: number;
      selected: boolean;
    };
    if (d.keys.length && it.kind !== "interactive") {
      // Only what reverts once the pointer leaves was caused by the hover.
      await page.mouse.move(0, 0);
      const still = (await page.evaluate(
        ([id, keys]) =>
          (
            window as unknown as {
              __hv: { recheck(i: number, k: string[]): string[] };
            }
          ).__hv.recheck(id as number, keys as string[]),
        [it.id, d.keys],
      )) as string[];
      d.changed = d.changed.filter((_, i) => !still.includes(d.keys[i]));
    }
    if (!d.hovered && it.kind !== "disabled") {
      stats.notHovered = (stats.notHovered ?? 0) + 1;
      continue;
    }
    const textField =
      /^(input|textarea)/.test(it.label) && !/checkbox|radio/.test(it.label);
    if (it.kind === "static") {
      if (d.changed.length)
        add("static-lights-up", it.label, d.changed.slice(0, 3).join("; "));
      if (d.cursor === "pointer")
        add(
          "static-pointer-cursor",
          it.label,
          "cursor:pointer on a non-interactive element",
        );
    } else if (it.kind === "disabled") {
      if (d.changed.length)
        add("disabled-lights-up", it.label, d.changed.slice(0, 3).join("; "));
      if (d.cursor === "pointer")
        add(
          "disabled-pointer-cursor",
          it.label,
          "cursor:pointer while disabled",
        );
    } else {
      // A hover that changes the control itself must ease (0.12 s), not snap.
      if (d.changed.some((c) => c.startsWith("self.")) && d.durMs === 0)
        add(
          "hover-snaps-no-transition",
          it.label,
          `${d.changed.find((c) => c.startsWith("self."))}; transition-duration 0s`,
        );
      // A borderless field that sits inside a surface of its own (the palette's search line) has nothing
      // to light up: the I-beam is its affordance.
      const bare = textField && /bg-transparent/.test(it.label);
      if (!d.changed.length && !d.selected && !bare)
        add(
          "interactive-no-hover-change",
          it.label,
          `${it.how}; cursor ${d.cursor}`,
        );
      // WebKit reports the UA-resolved cursor of links and text fields as "auto".
      const uaResolved = textField || /^a(\[|\.|$)/.test(it.label);
      const okCursor = uaResolved
        ? ["text", "pointer", "auto"].includes(d.cursor)
        : d.cursor === "pointer";
      if (!okCursor)
        add(
          "interactive-wrong-cursor",
          it.label,
          `${it.how}; cursor ${d.cursor}`,
        );
    }
  }
  await page.mouse.move(0, 0);

  if (overlay) {
    // An open popover traps focus: its tab stops are checked on the page-level run, not here.
    await page.close();
    return;
  }
  // Keyboard focus: every tab stop must show a ring.
  await page.evaluate(() => {
    (document.activeElement as HTMLElement | null)?.blur();
    (
      window as unknown as { __hv: { tagFocusables(): void } }
    ).__hv.tagFocusables();
  });
  await page.waitForTimeout(220);
  const seenIds = new Set<number>();
  let nulls = 0;
  for (let i = 0; i < 70 && nulls < 3; i++) {
    await page.keyboard.press("Tab");
    await page.waitForTimeout(25);
    const s = (await page.evaluate(() =>
      (
        window as unknown as {
          __hv: {
            focusState(): {
              id: number;
              label: string;
              focusVisible: boolean;
              ringBy: string;
              known: boolean;
              disabled: boolean;
            } | null;
          };
        }
      ).__hv.focusState(),
    )) as {
      id: number;
      label: string;
      focusVisible: boolean;
      ringBy: string;
      known: boolean;
      doubleRing: boolean;
    } | null;
    if (!s) {
      nulls += 1;
      continue;
    }
    nulls = 0;
    if (seenIds.has(s.id)) break;
    seenIds.add(s.id);
    stats[`${where}.tabStops`] = seenIds.size;
    if (!s.known) continue;
    if (!s.focusVisible)
      add(
        "focus-not-focus-visible",
        s.label,
        "Tab focus does not match :focus-visible",
      );
    else if (s.doubleRing)
      add(
        "focus-double-ring",
        s.label,
        "outline and box-shadow ring together (one ring per control)",
      );
    else if (!s.ringBy)
      add(
        "focus-no-ring",
        s.label,
        "no outline or ring change on keyboard focus",
      );
  }
  await page.close();
}

test("hover and focus affordance: interactive things respond, static things do not", async ({
  context,
  browser,
}, testInfo) => {
  test.setTimeout(150_000);
  await install(context);
  const out = new Map<string, Violation>();
  const stats: Record<string, number> = {};
  const overlays: Overlay[] = [
    {
      name: "settings+select",
      route: "settings",
      open: (p) =>
        p.getByRole("combobox", { name: "YUNSHU_LOG_LEVEL" }).click(),
    },
    {
      name: "requests+select",
      route: "requests",
      open: (p) => p.getByRole("combobox", { name: "分組方式" }).click(),
    },
    {
      name: "overview+notifications",
      route: "overview",
      open: (p) => p.getByRole("button", { name: "通知" }).click(),
    },
    {
      name: "overview+language",
      route: "overview",
      open: (p) => p.getByRole("button", { name: /^語言/ }).click(),
    },
    {
      name: "overview+palette",
      route: "overview",
      root: "[role=dialog]",
      open: async (p) => {
        await p.getByRole("button", { name: /^搜尋/ }).first().click();
        await p.getByRole("dialog").first().waitFor({ timeout: 10_000 });
      },
    },
    {
      name: "overview+island",
      route: "overview",
      root: "[role=dialog]",
      open: async (p) => {
        await p.getByTestId("footer-trigger").click();
        await p.getByRole("dialog").first().waitFor({ timeout: 10_000 });
      },
    },
  ];
  // Overlays morph open on animation frames, which a background tab does not run: each gets its own
  // context (its own foreground page).
  const solo = async (o: Overlay) => {
    const ctx = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      locale: "zh-TW",
    });
    try {
      await install(ctx);
      await auditPage(ctx, o.route, out, stats, o);
    } finally {
      await ctx.close();
    }
  };
  await Promise.all([
    ...PAGES.map((r) => auditPage(context, r, out, stats)),
    ...overlays.map(solo),
  ]);
  const all = [...out.values()].sort((a, b) =>
    (a.cat + a.label).localeCompare(b.cat + b.label),
  );
  const counts: Record<string, number> = {};
  for (const v of all) counts[v.cat] = (counts[v.cat] ?? 0) + 1;
  const lines = [
    `project=${testInfo.project.name}`,
    `counts=${JSON.stringify(counts)}`,
    `stats=${JSON.stringify(stats)}`,
    ...all.map(
      (v) => `[${v.cat}] ${v.label}  (${v.pages.join(",")})  ${v.detail}`,
    ),
    ...[...skipped].sort().map((k) => `[not-reachable] ${k}`),
  ];
  const text = lines.join("\n");
  if (process.env.HOVER_REPORT)
    writeFileSync(
      process.env.HOVER_REPORT.replace(
        /(\.\w+)?$/,
        `.${testInfo.project.name}$1`,
      ),
      text + "\n",
    );
  // Make sure the audit actually looked at something on every page.
  for (const o of overlays)
    expect(
      (stats[`${o.name}.interactive`] ?? 0) + (stats[`${o.name}.static`] ?? 0),
      `${o.name}: the overlay did not open or has nothing in it`,
    ).toBeGreaterThan(0);
  for (const r of PAGES)
    expect(
      (stats[`${r}.interactive`] ?? 0) + (stats[`${r}.static`] ?? 0),
      `${r}: nothing enumerated`,
    ).toBeGreaterThan(3);
  expect(all.length, text).toBe(0);
});

/** HOVER_SHOTS=/dir HOVER_SHOT_TAG=before|after: pictures of three hover states (not an assertion). */
test("hover screenshots", async ({ page, context }) => {
  test.skip(!process.env.HOVER_SHOTS, "set HOVER_SHOTS to write screenshots");
  const dir = process.env.HOVER_SHOTS!;
  const tag = process.env.HOVER_SHOT_TAG ?? "after";
  mkdirSync(dir, { recursive: true });
  await install(context);
  const shot = async (
    name: string,
    hover: () => Promise<void>,
    clip: () => Promise<{
      x: number;
      y: number;
      width: number;
      height: number;
    } | null>,
  ) => {
    await page.mouse.move(0, 0);
    await page.waitForTimeout(250);
    await hover();
    await page.waitForTimeout(300);
    const box = await clip();
    await page.screenshot({
      path: join(dir, `hover-${tag}-${name}-${test.info().project.name}.png`),
      ...(box ? { clip: box } : {}),
    });
  };
  const around = async (loc: ReturnType<Page["locator"]>, pad = 24) => {
    const b = await loc.boundingBox();
    if (!b) return null;
    const x = Math.max(0, b.x - pad);
    const y = Math.max(0, b.y - pad);
    return {
      x,
      y,
      width: Math.min(1440 - x, b.width + pad * 2),
      height: Math.min(900 - y, b.height + pad * 2),
    };
  };
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await page.waitForSelector("[data-stat-grid] .card");
  await page.waitForTimeout(1500);
  const stat = page.locator("[data-stat-grid] .card").first();
  await shot(
    "stat-card",
    () => stat.hover(),
    () => around(stat),
  );
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  await page.waitForSelector("tbody tr");
  await page.waitForTimeout(1500);
  const row = page.locator("tbody tr").first();
  await shot(
    "table-row",
    () => row.hover(),
    () => around(row, 12),
  );
  const tab = page.getByRole("tab").nth(1);
  await shot(
    "tab",
    () => tab.hover(),
    () => around(tab, 16),
  );
  const footer = page.getByTestId("footer-trigger");
  await shot(
    "footer-band",
    () => footer.hover({ position: { x: 40, y: 10 } }),
    () => around(footer, 12),
  );
  await page.goto("/console/#/settings", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(1500);
  const sw = page.getByRole("switch").first();
  await shot(
    "switch",
    () => sw.hover(),
    () => around(sw, 24),
  );
  const field = page.getByRole("textbox").first();
  await shot(
    "text-field",
    () => field.hover(),
    () => around(field, 16),
  );
});
