// Reads frontend/docs (MDX + meta.json) and builds what the console's 文件 section needs without
// compiling MDX: the navigation tree, every page's title / description / headings, and a plain-text
// body for search. Used by the Vite plugin (virtual modules), the link checker and the tests.
import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, relative } from "node:path";
import GithubSlugger from "github-slugger";

export const DOCS_DIR = new URL("../docs/", import.meta.url).pathname;
export const LOCALES = ["en", "zh-TW", "zh-CN"];
const SUFFIX = { en: "", "zh-TW": ".zh-TW", "zh-CN": ".zh-CN" };

/** `api/audio.zh-TW.mdx` -> { slug: "api/audio", locale: "zh-TW" }; null for other files. */
export function parseName(rel) {
  const m = /^(.*?)(?:\.(zh-TW|zh-CN))?\.mdx$/.exec(rel);
  return m ? { slug: m[1], locale: m[2] ?? "en" } : null;
}

export function walk(dir = DOCS_DIR) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...walk(p));
    else out.push(p);
  }
  return out;
}

/** All MDX files as { file, slug, locale }. */
export function listFiles(dir = DOCS_DIR) {
  return walk(dir)
    .map((file) => ({ file, ...(parseName(relative(dir, file)) ?? {}) }))
    .filter((f) => f.slug);
}

export function parseFrontmatter(text) {
  const m = /^---\n([\s\S]*?)\n---\n?/.exec(text);
  const meta = {};
  if (m)
    for (const line of m[1].split("\n")) {
      const kv = /^(\w+):\s*(.*)$/.exec(line);
      if (kv) meta[kv[1]] = kv[2].replace(/^["']|["']$/g, "");
    }
  return { meta, body: m ? text.slice(m[0].length) : text };
}

const plain = (s) =>
  s
    .replace(/`([^`]*)`/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, "")
    .replace(/[*_]{1,2}([^*_]+)[*_]{1,2}/g, "$1")
    .trim();

/** Headings (## to ####, outside code fences) with the ids rehype-slug gives them. */
export function headingsOf(body) {
  const slugger = new GithubSlugger();
  const out = [];
  let fence = false;
  for (const line of body.split("\n")) {
    if (/^\s*```/.test(line)) fence = !fence;
    if (fence) continue;
    const m = /^(#{1,4})\s+(.+?)\s*#*$/.exec(line);
    if (!m) continue;
    const text = plain(m[2]);
    const id = slugger.slug(text);
    if (m[1].length >= 2) out.push({ id, text, level: m[1].length });
  }
  return out;
}

/** Search text: prose and code kept, markup and component tags dropped. */
export function textOf(body) {
  return body
    .replace(/<\/?[A-Z][A-Za-z]*[^>]*>/g, " ")
    .replace(/^\s*\|?[\s:|-]+\|?\s*$/gm, " ")
    .replace(/[|#>*_`]/g, " ")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

export function readPage(file) {
  const { meta, body } = parseFrontmatter(readFileSync(file, "utf8"));
  return {
    title: meta.title ?? "",
    description: meta.description ?? "",
    headings: headingsOf(body),
    text: textOf(body),
  };
}

function readMeta(dir, locale) {
  for (const suffix of [SUFFIX[locale], ""]) {
    const f = join(dir, `meta${suffix}.json`);
    if (existsSync(f)) return JSON.parse(readFileSync(f, "utf8"));
  }
  return { pages: [] };
}

/** The file serving `slug` in `locale`: the locale's own, else the English one. */
export function pickFile(files, slug, locale) {
  return (
    files.find((f) => f.slug === slug && f.locale === locale) ??
    files.find((f) => f.slug === slug && f.locale === "en") ??
    null
  );
}

/**
 * Per-locale table of contents: { tree, pages }. `tree` follows the meta.json files (title + page
 * order per folder); `pages` maps slug -> { title, description, headings, locale } for every page.
 */
export function buildToc(locale, dir = DOCS_DIR) {
  const files = listFiles(dir);
  const pages = {};
  for (const slug of new Set(files.map((f) => f.slug))) {
    const f = pickFile(files, slug, locale);
    if (f) pages[slug] = { ...readPage(f.file), text: undefined, locale: f.locale };
  }
  const node = (folder) => {
    const m = readMeta(join(dir, folder), locale);
    const children = [];
    for (const name of m.pages ?? []) {
      const slug = folder ? `${folder}/${name}` : name;
      if (existsSync(join(dir, slug)) && statSync(join(dir, slug)).isDirectory())
        children.push(node(slug));
      else if (pages[slug]) children.push({ slug, title: pages[slug].title });
    }
    return { title: m.title ?? folder, slug: folder || null, children };
  };
  return { tree: node(""), pages };
}

/** Search entries for one locale: one per page, with its headings and plain text. */
export function buildSearch(locale, dir = DOCS_DIR) {
  const files = listFiles(dir);
  const out = [];
  for (const slug of new Set(files.map((f) => f.slug))) {
    const f = pickFile(files, slug, locale);
    if (!f) continue;
    const p = readPage(f.file);
    out.push({ slug, title: p.title, description: p.description, headings: p.headings, text: p.text });
  }
  return out;
}

/** Every `slug` in reading order (the tree flattened), for previous / next. */
export function flatten(tree) {
  const out = [];
  const go = (n) => {
    for (const c of n.children ?? []) {
      if (c.children) go(c);
      else out.push(c.slug);
    }
  };
  go(tree);
  return out;
}
