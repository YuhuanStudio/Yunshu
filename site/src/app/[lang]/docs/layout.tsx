import { DocsLayout } from "fumadocs-ui/layouts/docs";
import type { ReactNode } from "react";
import { source } from "@/lib/source";
import { isLang } from "@/lib/i18n";
import { Logo } from "@/components/logo";
import { ThemeToggle } from "@/components/theme-toggle";
import { REPO } from "@/lib/site";

export default async function DocsRootLayout(props: { children: ReactNode; params: Promise<{ lang: string }> }) {
  const { lang } = await props.params;
  if (!isLang(lang)) return null;
  return (
    <DocsLayout
      i18n
      tree={source.getPageTree(lang)}
      nav={{ title: <Logo suffix="Docs" />, url: `/${lang}` }}
      githubUrl={REPO}
      sidebar={{ defaultOpenLevel: 1, collapsible: true }}
      themeSwitch={{ component: <ThemeToggle /> }}
    >
      {props.children}
    </DocsLayout>
  );
}
