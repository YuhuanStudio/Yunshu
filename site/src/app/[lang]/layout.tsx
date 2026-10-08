import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import type { ReactNode } from "react";
import "../globals.css";
import { Providers } from "@/components/providers";
import { defineI18nUI } from "fumadocs-ui/i18n";
import { i18n } from "@/lib/i18n";
import { LANGS, LANG_NAMES, isLang, type Lang } from "@/lib/i18n";
import { asset } from "@/lib/site";
import { MESSAGES } from "@/lib/messages";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

const translations = Object.fromEntries(
  LANGS.map((l) => {
    const m = MESSAGES[l];
    return [
      l,
      {
        displayName: LANG_NAMES[l],
        "Search(search dialog)": m.search.replace("...", ""),
        "Search(search trigger)": m.search.replace("...", ""),
        "No results found(search dialog)": m.searchNoResult,
        "On this page(table of contents)": m.toc,
        "Table of Contents(inline table of contents)": m.toc,
        "Next Page(pagination)": m.next,
        "Previous Page(pagination)": m.previous,
        "Choose a language(language switcher)": m.language,
        "Choose a language(language switcher)(aria-label)": m.language,
        "Copy Markdown(page actions)": m.copyMd,
        "Copied Markdown(page actions)": m.copiedMd,
        "Open in GitHub(page actions)": m.openGithub,
        "Page Not Found(404 page)": m.notFound,
        "Back to Home(404 page)": m.backHome,
      },
    ];
  }),
) as Record<Lang, never>;

const { provider } = defineI18nUI(i18n, translations);

export const dynamicParams = false;

export function generateStaticParams() {
  return LANGS.map((lang) => ({ lang }));
}

export async function generateMetadata(props: { params: Promise<{ lang: string }> }): Promise<Metadata> {
  const { lang } = await props.params;
  const l: Lang = isLang(lang) ? lang : "en";
  return {
    title: { default: `Yunshu ${MESSAGES[l].docs}`, template: "%s | Yunshu" },
    description: MESSAGES[l].heroSubtitle,
    icons: { icon: asset("/favicon.ico"), apple: asset("/yuhuanstudio-logo.png") },
  };
}

export default async function LangLayout(props: { children: ReactNode; params: Promise<{ lang: string }> }) {
  const { lang } = await props.params;
  const l: Lang = isLang(lang) ? lang : "en";
  return (
    <html lang={l} suppressHydrationWarning data-scroll-behavior="smooth">
      <body className={`${geistSans.variable} ${geistMono.variable} flex min-h-screen flex-col antialiased`}>
        <Providers i18n={provider(l)}>{props.children}</Providers>
      </body>
    </html>
  );
}
