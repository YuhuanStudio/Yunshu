/**
 * Hash routes: `#/<page>[/<id>][?key=value&...]`.
 * The query carries intent, never state worth keeping: `#/models?action=load`
 * asks the Models page to open its load dialog, `#/logs?from=..&to=..` sets
 * the Logs window. Pages read their own keys with `routeQuery()`.
 */
export const PAGES = [
  "overview",
  "requests",
  "logs",
  "diagnostics",
  "models",
  "downloads",
  "cache",
  "playground",
  "api",
  "docs",
  "keys",
  "settings",
] as const;
export type Page = (typeof PAGES)[number];

/**
 * The console has six top-level pages. The rest are tabs of one of them and keep their own address,
 * so `#/downloads`, `#/logs`, `#/keys` ... still open (the old links keep working) and show their
 * parent's tab strip with the right tab selected.
 */
export const TABS: Partial<Record<Page, readonly Page[]>> = {
  models: ["models", "downloads", "cache"],
  diagnostics: ["diagnostics", "logs"],
  settings: ["settings", "keys", "api"],
};
export const PARENT: Partial<Record<Page, Page>> = {
  downloads: "models",
  cache: "models",
  logs: "diagnostics",
  keys: "settings",
  api: "settings",
};
/** The sidebar entry a page belongs to. */
export const topPage = (page: Page): Page => PARENT[page] ?? page;

export interface Route {
  page: Page;
  /** The part after the page (`#/models/<id>`, `#/docs/api/audio`), decoded; Models and Docs use it. */
  sub: string | null;
  query: URLSearchParams;
}

export function parseRoute(hash: string): Route {
  const body = hash.replace(/^#\/?/, "");
  const q = body.indexOf("?");
  const path = q === -1 ? body : body.slice(0, q);
  const query = new URLSearchParams(q === -1 ? "" : body.slice(q + 1));
  const [p = "", ...rest] = path.split("/");
  const page: Page = (PAGES as readonly string[]).includes(p)
    ? (p as Page)
    : "overview";
  let sub: string | null = null;
  try {
    sub =
      (page === "models" || page === "docs") && rest.length
        ? decodeURIComponent(rest.join("/"))
        : null;
  } catch {
    sub = null;
  }
  return { page, sub, query };
}

/** The current route's query, for pages that accept an intent such as `action=load`. */
export const routeQuery = (): URLSearchParams =>
  parseRoute(globalThis.location?.hash ?? "").query;

/** `#/<page>?k=v`, the link every palette verb and notification row uses. */
export function routeHref(
  page: Page,
  query?: Record<string, string>,
  sub?: string | null,
): string {
  const qs = query ? new URLSearchParams(query).toString() : "";
  return `#/${page}${sub ? "/" + encodeURIComponent(sub) : ""}${qs ? "?" + qs : ""}`;
}

/**
 * Palette verbs: each one is a deep link into the page that owns the action.
 * The page reads `action` (and `model`) from `routeQuery()` and opens its own
 * dialog; the shell never duplicates that logic.
 */
export const VERBS = [
  { id: "load", href: routeHref("models", { action: "load" }) },
  { id: "unload", href: routeHref("models", { action: "unload" }) },
  { id: "clearCache", href: routeHref("cache", { action: "clear" }) },
  { id: "download", href: routeHref("downloads", { action: "new" }) },
  { id: "createKey", href: routeHref("keys", { action: "create" }) },
  { id: "logs", href: routeHref("logs") },
] as const;
export type VerbId = (typeof VERBS)[number]["id"];

/** `g` then one key goes to a page (the Linear convention). */
export const CHORDS: Record<string, Page> = {
  o: "overview",
  r: "requests",
  l: "logs",
  d: "diagnostics",
  m: "models",
  w: "downloads",
  c: "cache",
  p: "playground",
  a: "api",
  f: "docs",
  k: "keys",
  s: "settings",
};

/** True when a key press belongs to a text field and must not trigger a shortcut. */
export function isTypingTarget(
  el: { tagName?: string; isContentEditable?: boolean } | null,
): boolean {
  if (!el) return false;
  const tag = (el.tagName ?? "").toLowerCase();
  return (
    el.isContentEditable === true ||
    tag === "input" ||
    tag === "textarea" ||
    tag === "select"
  );
}

/** The hash with the one-shot intent keys (`action`, `model`) removed, so a reload does not repeat them. */
export function withoutIntent(hash: string): string {
  const { page, sub, query } = parseRoute(hash);
  const keep: Record<string, string> = {};
  for (const [k, v] of query) if (k !== "action" && k !== "model") keep[k] = v;
  return routeHref(page, keep, sub);
}
