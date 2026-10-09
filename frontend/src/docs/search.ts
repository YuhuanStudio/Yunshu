import type { SearchEntry, SearchHit } from "./types.ts";

const words = (q: string) =>
  q
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 0);

function snippetAt(text: string, needle: string): string {
  const i = text.toLowerCase().indexOf(needle);
  if (i < 0) return text.slice(0, 120);
  let from = Math.max(0, i - 40);
  // start on a word boundary for space-separated text
  if (
    from > 0 &&
    /\S/.test(text[from - 1] ?? "") &&
    /[A-Za-z0-9]/.test(text[from] ?? "")
  ) {
    const sp = text.indexOf(" ", from);
    if (sp > 0 && sp < i) from = sp + 1;
  }
  let cut = text.slice(from, from + 130).trim();
  if (
    from + 130 < text.length &&
    /[A-Za-z0-9]$/.test(cut) &&
    /[A-Za-z0-9]/.test(text[from + 130] ?? "")
  )
    cut = cut.replace(/\s*\S*$/, "");
  return (from > 0 ? "… " : "") + cut + (from + 130 < text.length ? " …" : "");
}

/**
 * Plain search over the docs: every word of the query must appear in the page (title, headings,
 * description or text). Titles weigh most, then headings, then the description, then the body.
 * A heading hit points the result at that section.
 */
export function searchDocs(
  entries: SearchEntry[],
  query: string,
  limit = 8,
): SearchHit[] {
  const ws = words(query);
  if (!ws.length) return [];
  const scored: { score: number; hit: SearchHit }[] = [];
  for (const e of entries) {
    const title = e.title.toLowerCase();
    const desc = e.description.toLowerCase();
    const text = e.text.toLowerCase();
    let score = 0;
    let ok = true;
    let heading: SearchHit["heading"] = null;
    for (const w of ws) {
      const h = e.headings.find((x) => x.text.toLowerCase().includes(w));
      let s = 0;
      if (title.includes(w)) s += title === w ? 30 : 12;
      if (h) {
        s += 6;
        heading ??= h;
      }
      if (desc.includes(w)) s += 3;
      if (text.includes(w)) s += 1;
      if (!s) {
        ok = false;
        break;
      }
      score += s;
    }
    if (!ok) continue;
    const titleHit = ws.some((w) => title.includes(w));
    const snippetSource = e.description || e.text;
    scored.push({
      score,
      hit: {
        slug: e.slug,
        title: e.title,
        heading: titleHit ? null : heading,
        snippet: snippetAt(
          text.includes(ws[0]) ? e.text : snippetSource,
          ws[0],
        ),
      },
    });
  }
  return scored
    .sort((a, b) => b.score - a.score || a.hit.title.localeCompare(b.hit.title))
    .slice(0, limit)
    .map((s) => s.hit);
}
