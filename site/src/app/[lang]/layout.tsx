import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import type { ReactNode } from "react";
import "../globals.css";
import { Providers } from "@/components/providers";
import { defineI18nUI } from "fumadocs-ui/i18n";
import { LANGS, LANG_NAMES, i18n, isLang, type Lang } from "@/lib/i18n";
import { asset } from "@/lib/site";
import { MESSAGES } from "@/lib/messages";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

const { provider } = defineI18nUI(i18n, Object.fromEntries(
  LANGS.map((l) => {
    const m = MESSAGES[l];
    return [l, {
      displayName: LANG_NAMES[l],
      search: m.search,
      searchNoResult: m.searchNoResult,
      toc: m.toc,
      nextPage: m.next,
      previousPage: m.previous,
      chooseTheme: m.theme,
      chooseLanguage: m.language,
    }];
  }),
) as Record<Lang, never>);

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
