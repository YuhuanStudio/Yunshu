// Source-level link check for frontend/docs: every `/docs/<slug>` link (markdown or href=) must name
// an existing page, and every `#anchor` (own page or `/docs/<slug>#anchor`) a heading of that page in
// the page's own locale. Usage: node scripts/docs-links.mjs   (also run by tests/docs.test.ts)
import { readFileSync } from "node:fs";
import { relative } from "node:path";
import { DOCS_DIR, LOCALES, listFiles, parseFrontmatter, headingsOf } from "./docs-index.mjs";

const LINK = /\]\(([^)\s]+)\)|href="([^"]+)"/g;

export function checkLinks(dir = DOCS_DIR) {
  const files = listFiles(dir);
  const slugs = new Set(files.map((f) => f.slug));
  const ids = new Map();
  const idsOf = (slug, locale) => {
    const f = files.find((x) => x.slug === slug && x.locale === locale) ?? files.find((x) => x.slug === slug && x.locale === "en");
    const key = f.file;
    if (!ids.has(key)) ids.set(key, new Set(headingsOf(parseFrontmatter(readFileSync(key, "utf8")).body).map((h) => h.id)));
    return ids.get(key);
  };
  const broken = [];
  let checked = 0;
  for (const f of files) {
    const text = parseFrontmatter(readFileSync(f.file, "utf8")).body.replace(/^\s*```[\s\S]*?^\s*```/gm, "");
    for (const m of text.matchAll(LINK)) {
      const href = m[1] ?? m[2];
      if (/^(https?:|mailto:)/.test(href)) continue;
      const [path, hash] = href.split("#");
      let slug = f.slug;
      if (path) {
        const dm = /^\/docs\/?(.*?)\/?$/.exec(path);
        if (!dm) {
          broken.push(`${relative(dir, f.file)}: ${href} -> not a /docs link`);
          continue;
        }
        slug = dm[1] || "index";
      }
      checked++;
      if (!slugs.has(slug)) {
        broken.push(`${relative(dir, f.file)}: ${href} -> no such page`);
        continue;
      }
      if (hash && !idsOf(slug, f.locale).has(decodeURIComponent(hash)))
        broken.push(`${relative(dir, f.file)}: ${href} -> missing anchor #${hash}`);
    }
  }
  return { broken: [...new Set(broken)], checked, pages: files.length, locales: LOCALES.length };
}

if (process.argv[1] && process.argv[1].endsWith("docs-links.mjs")) {
  const { broken, checked, pages } = checkLinks();
  console.log(`checked ${checked} internal links in ${pages} pages`);
  if (broken.length) {
    console.error("Broken docs links:\n  " + broken.join("\n  "));
    process.exit(1);
  }
}
