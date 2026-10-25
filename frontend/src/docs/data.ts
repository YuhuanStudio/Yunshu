import { getBreadcrumbItems } from "fumadocs-core/breadcrumb";
import type { ComponentType } from "react";
import type { Locale } from "../i18n/index.ts";
import type { StaticSearch } from "./orama";
import type { Node } from "fumadocs-core/page-tree";
import type { DocsToc, TocNode } from "./types.ts";

const tocLoaders: Record<Locale, () => Promise<{ default: DocsToc }>> = {
  en: () => import("virtual:docs-toc-en"),
  "zh-TW": () => import("virtual:docs-toc-zh-TW"),
  "zh-CN": () => import("virtual:docs-toc-zh-CN"),
};
const searchLoaders: Record<Locale, () => Promise<{ default: StaticSearch }>> =
  {
    en: () => import("virtual:docs-search-en"),
    "zh-TW": () => import("virtual:docs-search-zh-TW"),
    "zh-CN": () => import("virtual:docs-search-zh-CN"),
  };
const tocs = new Map<Locale, Promise<DocsToc>>();
const searches = new Map<Locale, Promise<StaticSearch>>();

export function loadToc(locale: Locale): Promise<DocsToc> {
  let p = tocs.get(locale);
  if (!p) {
    p = Promise.all([tocLoaders[locale](), import("./source")]).then(
      ([m, { source }]) => {
        const convert = (node: Node): TocNode[] => {
          if (node.type === "separator") return [];
          if (node.type === "page")
            return [
              {
                slug: node.url.replace(/^\/docs\/?/, "") || "index",
                title: String(node.name),
              },
            ];
          return [
            {
              slug: node.$ref?.folder ?? String(node.name),
              title: String(node.name),
              children: [
                ...(node.index ? convert(node.index) : []),
                ...node.children.flatMap(convert),
              ],
            },
          ];
        };
        const pageTree = source.getPageTree(locale);
        return {
          ...m.default,
          pages: Object.fromEntries(
            Object.entries(m.default.pages).map(([slug, info]) => [
              slug,
              {
                ...info,
                breadcrumbs: getBreadcrumbItems(
                  slug === "index" ? "/docs" : `/docs/${slug}`,
                  pageTree,
                ).map((item) => ({ title: String(item.name), url: item.url })),
              },
            ]),
          ),
          tree: {
            title: "Docs",
            slug: null,
            children: pageTree.children.flatMap(convert),
          },
        };
      },
    );
    tocs.set(locale, p);
    p.catch(() => tocs.delete(locale));
  }
  return p;
}

export function loadSearch(locale: Locale): Promise<StaticSearch> {
  let p = searches.get(locale);
  if (!p) {
    p = searchLoaders[locale]().then((m) => m.default);
    searches.set(locale, p);
    p.catch(() => searches.delete(locale));
  }
  return p;
}

type PageModule = {
  default: ComponentType<{ components?: import("mdx/types").MDXComponents }>;
};
/** Fumadocs MDX owns compilation, source metadata and lazy page imports. */
export function pageLoader(
  slug: string,
  locale: Locale,
): () => Promise<PageModule> {
  return async () => {
    const { collection } = await import("./source");
    const suffix = locale === "en" ? "" : "." + locale;
    const entry =
      collection.getPage(`${slug}${suffix}.mdx`) ??
      collection.getPage(`${slug}.mdx`);
    if (!entry) throw new Error(`Unknown document: ${slug}`);
    await entry.preload();
    return { default: entry.body };
  };
}

export { docHref, resolveDocLink, slugOf } from "./links.ts";
