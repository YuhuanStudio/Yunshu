import { source } from "@/lib/source";
import { DocsPage, DocsBody, DocsDescription, DocsTitle } from "fumadocs-ui/layouts/docs/page";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { getMDXComponents } from "../../../../../mdx-components";
import { createCard, createLink } from "@/components/localized-link";
import { Feedback } from "@/components/feedback";
import { PageActions } from "@/components/page-actions";
import { isLang } from "@/lib/i18n";
import { MESSAGES } from "@/lib/messages";
import { REPO, asset } from "@/lib/site";

export const dynamicParams = false;

export default async function Page(props: { params: Promise<{ lang: string; slug?: string[] }> }) {
  const { lang, slug } = await props.params;
  if (!isLang(lang)) notFound();
  const page = source.getPage(slug, lang);
  if (!page) notFound();
  const m = MESSAGES[lang];
  const Mdx = page.data.body;
  const mdUrl = asset(`/${lang}/raw/${slug?.length ? slug.join("/") : "index"}`);

  return (
    <DocsPage toc={page.data.toc} full={page.data.full} tableOfContent={{ style: "clerk", single: false }}>
      <DocsTitle>{page.data.title}</DocsTitle>
      <DocsDescription>{page.data.description}</DocsDescription>
      <PageActions
        messages={m}
        markdownUrl={mdUrl}
        githubUrl={`${REPO}/blob/main/site/content/docs/${page.path}`}
      />
      <DocsBody>
        <Mdx components={getMDXComponents({ a: createLink(lang), Card: createCard(lang) })} />
      </DocsBody>
      <Feedback messages={m} path={page.url} />
    </DocsPage>
  );
}

export function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(props: { params: Promise<{ lang: string; slug?: string[] }> }): Promise<Metadata> {
  const { lang, slug } = await props.params;
  const page = source.getPage(slug, lang);
  if (!page) notFound();
  return { title: page.data.title, description: page.data.description };
}
