import { useSyncExternalStore } from "react";
import zhTW from "./locales/zh-TW/index.ts";
import { interpolate } from "./plural.ts";
import {
  DEFAULT_LOCALE,
  HTML_LANG,
  LOCALES,
  type Key,
  type Locale,
  type Vars,
} from "./types.ts";

export { LOCALES, LOCALE_NAMES, type Key, type Locale } from "./types.ts";

const STORAGE_KEY = "yunshu.console.locale";

type Flat = Record<string, string>;
function flatten(dict: Record<string, Record<string, string>>): Flat {
  const out: Flat = {};
  for (const [ns, entries] of Object.entries(dict))
    for (const [k, v] of Object.entries(entries)) out[`${ns}.${k}`] = v;
  return out;
}

const flats = new Map<Locale, Flat>([["zh-TW", flatten(zhTW)]]);
const loaders: Record<
  Exclude<Locale, "zh-TW">,
  () => Promise<{ default: object }>
> = {
  "zh-CN": () => import("./locales/zh-CN/index.ts"),
  en: () => import("./locales/en/index.ts"),
};

export const isLocale = (v: unknown): v is Locale =>
  typeof v === "string" && (LOCALES as readonly string[]).includes(v);

/** Map one BCP 47 tag onto a supported locale; Hong Kong and Macau read Traditional. */
export function matchLocale(tag: string): Locale | null {
  const t = tag.toLowerCase();
  if (/^zh-(tw|hk|mo|hant)/.test(t)) return "zh-TW";
  if (/^zh-(cn|sg|hans)/.test(t)) return "zh-CN";
  if (t === "zh") return "zh-TW";
  if (t === "en" || t.startsWith("en-")) return "en";
  return null;
}

/** Resolution: ?lang= > saved choice > navigator.languages > zh-TW. */
export function detectLocale(): Locale {
  try {
    const q = new URLSearchParams(location.search).get("lang");
    const m = q ? matchLocale(q) : null;
    if (m) return m;
  } catch {}
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (isLocale(saved)) return saved;
  } catch {}
  try {
    const langs = navigator.languages?.length
      ? navigator.languages
      : [navigator.language];
    for (const tag of langs) {
      const m = tag ? matchLocale(tag) : null;
      if (m) return m;
    }
  } catch {}
  return DEFAULT_LOCALE;
}

let locale: Locale = DEFAULT_LOCALE;
const listeners = new Set<() => void>();

export const getLocale = () => locale;

async function load(next: Locale) {
  if (flats.has(next) || next === "zh-TW") return;
  const mod = (await loaders[next]()).default as Record<
    string,
    Record<string, string>
  >;
  flats.set(next, flatten(mod));
}

function apply(next: Locale) {
  locale = next;
  try {
    document.documentElement.lang = HTML_LANG[next];
  } catch {}
  listeners.forEach((l) => l());
}

/** Load the detected locale before the first render, so there is no flash of another language. */
export async function initI18n(): Promise<void> {
  const next = detectLocale();
  try {
    await load(next);
    apply(next);
  } catch {
    apply(DEFAULT_LOCALE);
  }
}

/** Switch language without a reload. Failing to load a chunk keeps the current language. */
export async function setLocale(next: Locale): Promise<void> {
  if (next === locale) return;
  try {
    await load(next);
  } catch {
    return;
  }
  try {
    localStorage.setItem(STORAGE_KEY, next);
  } catch {}
  apply(next);
}

/** Subscribe a component to locale changes; returns the active locale. */
export function useLocale(): Locale {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    getLocale,
    getLocale,
  );
}

/** Untyped lookup for keys assembled at runtime; prefer {@link t}. */
export function tr(key: string, vars?: Vars): string {
  const msg = flats.get(locale)?.[key] ?? flats.get("zh-TW")![key];
  // A visible marker makes a missing key obvious in tests; production never sees one.
  if (msg == null) return `⟦${key}⟧`;
  return interpolate(msg, vars, locale);
}

/** Typed lookup: a misspelled key is a tsc error. */
export function t(key: Key, vars?: Vars): string {
  return tr(key, vars);
}

/** Whether a key exists in the active dictionary (used by YunUI's fallback logic). */
export function has(key: string): boolean {
  return flats.get(locale)?.[key] != null;
}
