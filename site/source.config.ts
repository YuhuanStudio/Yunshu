import { defineDocs, defineConfig } from "fumadocs-mdx/config";

export const docs = defineDocs({
  dir: "content/docs",
  docs: { postprocess: { includeProcessedMarkdown: true } },
});

export default defineConfig({
  mdxOptions: {
    // GitHub's accessibility-corrected palettes, as YunUI's CodeBlock uses.
    rehypeCodeOptions: { themes: { light: "github-light-default", dark: "github-dark-default" } },
  },
});
