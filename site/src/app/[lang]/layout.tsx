import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import type { ReactNode } from "react";
import "../globals.css";
import { Providers } from "@/components/providers";
import { LANGS, isLang, type Lang } from "@/lib/i18n";
import { asset } from "@/lib/site";
import { MESSAGES } from "@/lib/messages";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

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
        <Providers lang={l}>{props.children}</Providers>
      </body>
    </html>
  );
}
