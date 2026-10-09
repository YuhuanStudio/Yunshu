import type { ComponentType } from "react";
import type { Locale } from "../i18n/index.ts";
import type { DocsToc, SearchEntry } from "./types.ts";

const tocLoaders: Record<Locale, () => Promise<{ default: DocsToc }>> = {
  en: () => import("virtual:docs-toc-en"),
  "zh-TW": () => import("virtual:docs-toc-zh-TW"),
  "zh-CN": () => import("virtual:docs-toc-zh-CN"),
};
const searchLoaders: Record<Locale, () => Promise<{ default: SearchEntry[] }>> =
  {
    en: () => import("virtual:docs-search-en"),
    "zh-TW": () => import("virtual:docs-search-zh-TW"),
    "zh-CN": () => import("virtual:docs-search-zh-CN"),
  };
const tocs = new Map<Locale, Promise<DocsToc>>();
const searches = new Map<Locale, Promise<SearchEntry[]>>();

export function loadToc(locale: Locale): Promise<DocsToc> {
  let p = tocs.get(locale);
  if (!p) {
    p = tocLoaders[locale]().then((m) => m.default);
    tocs.set(locale, p);
    p.catch(() => tocs.delete(locale));
  }
  return p;
}

export function loadSearch(locale: Locale): Promise<SearchEntry[]> {
  let p = searches.get(locale);
  if (!p) {
    p = searchLoaders[locale]().then((m) => m.default);
    searches.set(locale, p);
    p.catch(() => searches.delete(locale));
  }
  return p;
}

type PageModule = { default: ComponentType<{ components?: object }> };
const pages = import.meta.glob<PageModule>("../../docs/**/*.mdx");

/** Loader of one page in the console language, else its English source. */
export function pageLoader(
  slug: string,
  locale: Locale,
): (() => Promise<PageModule>) | null {
  const suffix = locale === "en" ? "" : "." + locale;
  return (
    pages[`../../docs/${slug}${suffix}.mdx`] ??
    pages[`../../docs/${slug}.mdx`] ??
    null
  );
}

export { docHref, resolveDocLink, slugOf } from "./links.ts";
