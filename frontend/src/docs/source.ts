import { defineDocs } from "fumadocs-mdx/macro";
import { loader } from "fumadocs-core/source";

export const collection = defineDocs({
  dir: "docs",
  docs: { files: ["**/*.mdx"], async: true, lastModified: true },
});

export const source = loader({
  baseUrl: "/docs",
  url: (slugs) => "/docs/" + slugs.join("/"),
  source: collection.toFumadocsSource(),
  i18n: { defaultLanguage: "en", languages: ["en", "zh-TW", "zh-CN"] },
});
