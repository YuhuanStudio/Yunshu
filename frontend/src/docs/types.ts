export interface Heading {
  id: string;
  text: string;
  level: number;
}
export interface PageInfo {
  lastUpdated?: string;
  breadcrumbs?: { title: string; url?: string }[];
  title: string;
  description: string;
  headings: Heading[];
  /** The language the page is actually written in (falls back to English). */
  locale: string;
}
export interface TocNode {
  title: string;
  slug: string | null;
  children?: TocNode[];
}
export interface DocsToc {
  tree: TocNode;
  /** The fumadocs page tree of this locale, which the layout renders as the sidebar. */
  pageTree: import("fumadocs-core/page-tree").Root;
  pages: Record<string, PageInfo>;
}
export interface SearchEntry {
  slug: string;
  title: string;
  description: string;
  headings: Heading[];
  text: string;
}
export interface SearchHit {
  slug: string;
  title: string;
  /** The heading the match sits under, when it is not the page title. */
  heading: Heading | null;
  snippet: string;
}
