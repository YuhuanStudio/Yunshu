"use client";

import { ThemeProvider } from "next-themes";
import { RootProvider } from "fumadocs-ui/provider/next";
import { YunUIProvider } from "@yuhuanowo/yunui/adapters";
import NextLink from "next/link";
import { useRouter as useNextRouter } from "next/navigation";
import type { ComponentProps, ReactNode } from "react";
import type { I18nProviderProps } from "fumadocs-ui/contexts/i18n";
import { BASE_PATH } from "@/lib/site";

export function Providers({ i18n, children }: { i18n: Omit<I18nProviderProps, "children" | "onLocaleChange">; children: ReactNode }) {
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
          ...i18n,
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
