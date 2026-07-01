"use client";

import type { ReactNode } from "react";
import { ThemeProvider } from "next-themes";
import NextLink from "next/link";
import NextImage from "next/image";
import { useRouter as useNextRouter } from "next/navigation";
import { YunUIProvider } from "yunui/adapters";
import { Toaster } from "yunui";

/**
 * App-wide providers for the Yunshu webui.
 *
 * - `next-themes` owns the theme class/attribute on <html> (light / dark /
 *   true-black / system), which is what YunUI's design tokens key off. Setting
 *   both `class` and `data-theme` keeps YunUI's dark styles and the markdown
 *   prose overrides (`[data-theme="light"]`) in sync.
 * - `YunUIProvider` injects Next's Link / Image / router so YunUI's
 *   framework-coupled components navigate client-side and optimize images.
 *
 * Yunshu has no i18n library, so `useT` returns English labels for the handful
 * of component strings YunUI asks for (mainly the ThemeToggle menu) and falls
 * back to the bare key for everything else.
 */

const STRINGS: Record<string, string> = {
  "common.theme.toggle": "Toggle theme",
  "common.theme.light": "Light",
  "common.theme.zincDark": "Dark",
  "common.theme.trueBlack": "True black",
  "common.theme.system": "System",
};

const useT = (namespace?: string) => (key: string) => {
  const full = namespace ? `${namespace}.${key}` : key;
  return STRINGS[full] ?? key;
};

export function Providers({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider
      attribute={["class", "data-theme"]}
      defaultTheme="dark"
      enableSystem
      themes={["light", "dark", "true-black"]}
      value={{ light: "light", dark: "dark", "true-black": "true-black" }}
    >
      <YunUIProvider
        adapters={{
          Link: NextLink as never,
          Image: NextImage as never,
          useRouter: () => {
            const r = useNextRouter();
            return { push: r.push, replace: r.replace, back: r.back };
          },
          useT,
          iconBasePath: "/icons",
        }}
      >
        {children}
        <Toaster />
      </YunUIProvider>
    </ThemeProvider>
  );
}
