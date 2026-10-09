"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ArrowRight, BookOpen, Code2, Rocket, Wrench } from "lucide-react";
import { Footer, GithubIcon, LanguageSwitcher, Navbar } from "@yuhuanowo/yunui/ai";
import { CodeBlock, Eyebrow, FeatureCard, HeroAccent, MarketingHero } from "@yuhuanowo/yunui/patterns";
import { LANGS, LANG_NAMES, type Lang } from "@/lib/i18n";
import { MESSAGES } from "@/lib/messages";
import { BASE_PATH, REPO, asset } from "@/lib/site";
import { Logo } from "@/components/logo";
import { ThemeToggle } from "@/components/theme-toggle";

const CODE = `uv tool install "yunshu[vision]" --python 3.13
yunshu pull mlx-community/Qwen3.5-0.8B-MLX-bf16
yunshu serve -m mlx-community/Qwen3.5-0.8B-MLX-bf16`;

const ICONS = [Rocket, Code2, BookOpen, Wrench];

export function Home({ lang }: { lang: Lang }) {
  const m = MESSAGES[lang];
  const pathname = usePathname();
  const p = (href: string) => `/${lang}${href}`;

  const switchLang = (next: string) => {
    try {
      localStorage.setItem("yunshu-docs-lang", next);
    } catch {
      /* private mode */
    }
    window.location.assign(`${BASE_PATH}/${next}/`);
  };

  return (
    <>
      <Navbar
        appName="Yunshu"
        brand={<Logo />}
        homeHref={`/${lang}`}
        links={[
          { label: m.docs, href: p("/docs/getting-started/install") },
          { label: m.api, href: p("/docs/api/overview") },
          { label: m.guides, href: p("/docs/guides/agents") },
          { label: m.developers, href: p("/docs/developers/architecture") },
        ]}
        currentPath={pathname}
        labels={{ menu: m.menu }}
        languageSwitcher={
          <LanguageSwitcher
            variant="pill"
            locales={LANGS.map((l) => ({ value: l, label: LANG_NAMES[l] }))}
            currentLocale={lang}
            onChange={switchLang}
            label={m.language}
          />
        }
        themeToggle={<ThemeToggle />}
        actions={
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer noopener"
            aria-label={m.github}
            className="text-muted-foreground hover:text-foreground hover:bg-foreground/5 flex h-9 w-9 items-center justify-center rounded-full transition-colors"
          >
            <GithubIcon />
          </a>
        }
        account={<></>}
      />
      <main className="flex-1">
        <MarketingHero
          fullHeight={false}
          badge={<Eyebrow>{m.heroBadge}</Eyebrow>}
          title={
            <>
              {m.heroTitle} <HeroAccent>{m.heroAccent}</HeroAccent>
            </>
          }
          subtitle={m.heroSubtitle}
          actions={
            <div className="flex flex-wrap items-center justify-center gap-3">
              <Link
                href={p("/docs/getting-started/install")}
                className="bg-foreground text-background inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium transition-opacity hover:opacity-90"
              >
                {m.start}
                <ArrowRight className="size-4" aria-hidden />
              </Link>
              <Link
                href={p("/docs/api/overview")}
                className="border-border hover:bg-foreground/5 inline-flex items-center gap-2 rounded-full border px-5 py-2.5 text-sm font-medium transition-colors"
              >
                {m.apiRef}
              </Link>
            </div>
          }
          facts={m.facts}
        />
        <section className="mx-auto w-full max-w-5xl px-4 pb-16 sm:px-6">
          <div className="grid gap-4 sm:grid-cols-2">
            {m.sections.map((s, i) => {
              const Icon = ICONS[i];
              return (
                <Link key={s.href} href={p(s.href)} className="block rounded-2xl focus-visible:outline-2">
                  <FeatureCard icon={<Icon className="size-5" aria-hidden />} title={s.title} description={s.desc} className="h-full" />
                </Link>
              );
            })}
          </div>
        </section>
        <section className="mx-auto w-full max-w-3xl px-4 pb-20 sm:px-6">
          <h2 className="mb-4 text-center text-xl font-semibold tracking-tight">{m.codeTitle}</h2>
          <CodeBlock code={CODE} language="bash" />
        </section>
      </main>
      <Footer
        appName="Yunshu"
        logoSrc={asset("/yuhuanstudio-logo.png")}
        homeHref={`/${lang}`}
        tagline={m.footer}
        social={[{ icon: <GithubIcon />, href: REPO, label: m.github }]}
        copyright="YuhuanStudio"
      />
    </>
  );
}
