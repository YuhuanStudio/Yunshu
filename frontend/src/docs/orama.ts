import { create, load, search } from "@orama/orama";
import { tokenizeDocs } from "./orama-tokenizer.ts";
import type { SearchEntry, SearchHit } from "./types.ts";
import { searchDocs } from "./search.ts";

export const docsSchema = {
  slug: "string",
  title: "string",
  description: "string",
  text: "string",
  headings: "string",
} as const;
export function createDocsDatabase() {
  return create({
    schema: docsSchema,
    components: { tokenizer: { language: "english", tokenize: tokenizeDocs } },
  });
}
export type StaticSearch = { data: unknown; entries: SearchEntry[] };
const databases = new WeakMap<
  StaticSearch,
  ReturnType<typeof createDocsDatabase>
>();
export async function queryDocs(
  index: StaticSearch,
  query: string,
  limit = 8,
): Promise<SearchHit[]> {
  if (!query.trim()) return [];
  let db = databases.get(index);
  if (!db) {
    db = createDocsDatabase();
    load(db, index.data as Parameters<typeof load>[1]);
    databases.set(index, db);
  }
  const result = await search(db, {
    term: query,
    properties: ["title", "description", "text", "headings"],
    boost: { title: 6, headings: 3, description: 2 },
    // Retrieve the indexed union, then apply the console's all-words filter
    // across fields. Orama's strict threshold counts prefix expansions, which
    // drops CJK character+bigram matches and words split across properties.
    threshold: 1,
    limit: index.entries.length,
  });
  // Keep the console result shape and section-aware snippets; Orama supplies
  // indexed candidate retrieval instead of scanning every page per keystroke.
  const candidates = result.hits
    .map((h) => index.entries.find((e) => e.slug === h.document.slug)!)
    .filter(Boolean);
  return searchDocs(candidates, query, limit);
}
