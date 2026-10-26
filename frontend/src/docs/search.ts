import type { SearchEntry, SearchHit } from "./types.ts";

import { splitHanRuns, STOP_WORDS } from "./orama-tokenizer.ts";

const HAN = /\p{Script=Han}/u;
/** Query words: whitespace words minus English function words; a Han run becomes its
 * content bigrams (a question like "如何安裝" is not one literal substring). */
export function queryWords(q: string): { words: string[]; han: string[] } {
  const words: string[] = [];
  const han: string[] = [];
  for (const w of q.toLowerCase().split(/\s+/)) {
    if (!w) continue;
    if (!HAN.test(w)) {
      const word = w.replace(/^[^a-z0-9_]+|[^a-z0-9_]+$/g, "");
      if (word && !STOP_WORDS.has(word)) words.push(word);
      continue;
    }
    for (const run of splitHanRuns(w.replace(/[^\p{Script=Han}]/gu, ""))) {
      const chars = Array.from(run);
      if (chars.length === 1) han.push(run);
      for (let i = 0; i + 1 < chars.length; i++)
        han.push(chars[i] + chars[i + 1]);
    }
  }
  return { words, han: [...new Set(han)] };
}

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
  const parsed = queryWords(query);
  // Han bigrams score like words, but a question only needs most of them to appear.
  const ws = [...parsed.words, ...parsed.han];
  if (!ws.length) return [];
  const hanNeeded = Math.ceil(0.6 * parsed.han.length);
  const scored: { score: number; hit: SearchHit }[] = [];
  for (const e of entries) {
    const title = e.title.toLowerCase();
    const desc = e.description.toLowerCase();
    const text = e.text.toLowerCase();
    let score = 0;
    let ok = true;
    let hanHits = 0;
    let heading: SearchHit["heading"] = null;
    for (const w of ws) {
      const isHan = parsed.han.includes(w);
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
        if (isHan) continue;
        ok = false;
        break;
      }
      if (isHan) hanHits++;
      score += s;
    }
    if (!ok || hanHits < hanNeeded) continue;
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
