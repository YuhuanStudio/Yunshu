// Heading-level retrieval artifacts shared by the console backend and Orama tests.
import { readFileSync } from "node:fs";
import GithubSlugger from "github-slugger";
import {
  listFiles,
  pickFile,
  parseFrontmatter,
  textOf,
  DOCS_DIR,
  plain,
} from "./docs-index.mjs";

export function buildChunks(locale, dir = DOCS_DIR) {
  const files = listFiles(dir);
  const chunks = [];
  for (const slug of new Set(files.map((f) => f.slug))) {
    const file = pickFile(files, slug, locale);
    if (!file) continue;
    const { meta, body } = parseFrontmatter(readFileSync(file.file, "utf8"));
    const slugger = new GithubSlugger();
    let anchor = "",
      heading = "",
      lines = [],
      fence = null;
    const flush = () => {
      const text = lines.join("\n").trim();
      if (!text) return;
      const url = `/docs${slug === "index" ? "" : "/" + slug}${anchor ? "#" + anchor : ""}`;
      chunks.push({
        id: url,
        slug,
        url,
        anchor,
        title: meta.title ?? slug,
        description: meta.description ?? "",
        heading,
        text,
        searchText: textOf(text),
      });
    };
    for (const line of body.split("\n")) {
      const mark = /^\s*(`{3,}|~{3,})/.exec(line)?.[1];
      if (mark && !fence) fence = mark[0];
      else if (mark && fence === mark[0]) {
        fence = null;
        lines.push(line);
        continue;
      }
      const m = !fence && /^(#{1,6})\s+(.+?)\s*#*$/.exec(line);
      if (m) {
        flush();
        lines = [];
        heading = plain(m[2]);
        anchor = slugger.slug(heading);
      }
      lines.push(line);
    }
    flush();
  }
  return { version: 1, locale, chunks };
}
