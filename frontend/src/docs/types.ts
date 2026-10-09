export interface Heading {
  id: string;
  text: string;
  level: number;
}
export interface PageInfo {
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
