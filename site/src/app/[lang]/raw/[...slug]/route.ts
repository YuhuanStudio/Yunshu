import { getLLMText, source } from "@/lib/source";
import { notFound } from "next/navigation";

export const dynamic = "force-static";
export const dynamicParams = false;

// /<lang>/raw/<page path> serves the page's Markdown ("Copy Markdown"); the docs index is `index`.
export async function GET(_req: Request, { params }: { params: Promise<{ lang: string; slug: string[] }> }) {
  const { lang, slug } = await params;
  const page = source.getPage(slug.length === 1 && slug[0] === "index" ? undefined : slug, lang);
  if (!page) notFound();
  return new Response(await getLLMText(page), { headers: { "Content-Type": "text/markdown; charset=utf-8" } });
}

export function generateStaticParams() {
  return source.generateParams().map((p) => ({ lang: p.lang, slug: p.slug?.length ? p.slug : ["index"] }));
}
