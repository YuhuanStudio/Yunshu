"use client";

import { ThemeProvider } from "next-themes";
import { RootProvider } from "fumadocs-ui/provider/next";
import { YunUIProvider } from "@yuhuanowo/yunui/adapters";
import NextLink from "next/link";
import { useRouter as useNextRouter } from "next/navigation";
import type { ComponentProps, ReactNode } from "react";
import { LANGS, LANG_NAMES, type Lang } from "@/lib/i18n";
import { MESSAGES } from "@/lib/messages";
import { BASE_PATH } from "@/lib/site";

export function Providers({ lang, children }: { lang: Lang; children: ReactNode }) {
  const m = MESSAGES[lang];
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      themes={["light", "dark", "true-black"]}
      value={{ light: "light", dark: "dark", "true-black": "true-black" }}
    >
      <RootProvider
        // next-themes above owns theming (light/dark/true-black).
        theme={{ enabled: false }}
        search={{ options: { type: "static", api: `${BASE_PATH}/api/search` } }}
        i18n={{
          locale: lang,
          locales: LANGS.map((l) => ({ locale: l, name: LANG_NAMES[l] })),
          translations: {
            search: m.search,
            searchNoResult: m.searchNoResult,
            toc: m.toc,
            nextPage: m.next,
            previousPage: m.previous,
            chooseTheme: m.theme,
            chooseLanguage: m.language,
          },
          onLocaleChange(next: string) {
            try {
              localStorage.setItem("yunshu-docs-lang", next);
            } catch {
              /* private mode */
            }
            const path = window.location.pathname.replace(BASE_PATH, "");
            const rest = path.replace(/^\/(en|zh-TW|zh-CN)(?=\/|$)/, "");
            window.location.assign(`${BASE_PATH}/${next}${rest || "/"}`);
          },
        }}
      >
        <YunUIProvider
          adapters={{
            Link: NextLink as never,
            useRouter: () => {
              const r = useNextRouter();
              return { push: r.push, replace: r.replace, back: r.back };
            },
            useT: () => (key: string) => key,
          }}
        >
          {children}
        </YunUIProvider>
      </RootProvider>
    </ThemeProvider>
  );
}

export type LinkProps = ComponentProps<typeof NextLink>;
