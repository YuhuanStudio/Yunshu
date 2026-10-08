import { docs } from "fumadocs-mdx:collections/server";
import { loader, type InferPageType } from "fumadocs-core/source";
import { i18n } from "@/lib/i18n";

export const source = loader({
  baseUrl: "/docs",
  source: docs.toFumadocsSource(),
  i18n,
});

export async function getLLMText(page: InferPageType<typeof source>) {
  const processed = await page.data.getText("processed");
  return `# ${page.data.title} (${page.url})\n\n${processed}`;
}
