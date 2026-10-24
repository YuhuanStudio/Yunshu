import {
  Suspense,
  lazy,
  useEffect,
  useMemo,
  useState,
  type ComponentType,
} from "react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
  Spinner,
} from "@yuhuanowo/yunui";
import { DashboardPage } from "@yuhuanowo/yunui/patterns";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { AnchorProvider, TOCItem } from "fumadocs-core/toc";
import { DocsProvider } from "./docs/provider";
import { PageActions } from "./docs/PageActions";
import "./docs/fumadocs.css";
import { t, useLocale, type Locale } from "./i18n/index.ts";
import { routeQuery } from "./route";
import { setDocTitle } from "./docs/title-store.ts";
import {
  docHref,
  resolveDocLink,
  loadToc,
  pageLoader,
  slugOf,
} from "./docs/data.ts";
import { mdxComponents } from "./docs/mdx-components";
import type { DocsToc, TocNode } from "./docs/types.ts";

const ApiReference = lazy(() => import("./docs/ApiReference"));

type Loaded = {
  slug: string;
  locale: Locale;
  Page: ComponentType<{ components?: import("mdx/types").MDXComponents }>;
};

const leaves = (n: TocNode): string[] =>
  (n.children ?? []).flatMap((c) => (c.children ? leaves(c) : [c.slug!]));

function Tree({
  node,
  slug,
  depth = 0,
}: {
  node: TocNode;
  slug: string;
  depth?: number;
}) {
  return (
    <ul className="space-y-0.5">
      {(node.children ?? []).map((c) =>
        c.children ? (
          <li key={c.slug} className="pt-3">
            <p className="px-2 pb-1 text-xs font-medium text-muted-foreground">
              {c.title}
            </p>
            <Tree node={c} slug={slug} depth={depth + 1} />
          </li>
        ) : (
          <li key={c.slug}>
            <a
              href={docHref(c.slug!)}
              aria-current={c.slug === slug ? "page" : undefined}
              className={`block rounded-lg px-2 py-1.5 text-sm transition-colors hover:bg-(--bg-elevated) ${
                c.slug === slug
                  ? "bg-(--bg-elevated) font-medium text-foreground"
                  : "text-muted-foreground"
              }`}
            >
              {c.title}
            </a>
          </li>
        ),
      )}
    </ul>
  );
}

function PageJump({ toc, slug }: { toc: DocsToc; slug: string }) {
  return (
    <Select
      value={slug}
      onValueChange={(v) => {
        location.hash = docHref(v).slice(1);
      }}
    >
      <SelectTrigger aria-label={t("docs.nav.jump")}>
        <SelectValue placeholder={t("docs.nav.jump")} />
      </SelectTrigger>
      <SelectContent>
        {(toc.tree.children ?? []).map((n) =>
          n.children ? (
            <SelectGroup key={n.slug}>
              <SelectLabel>{n.title}</SelectLabel>
              {leaves(n).map((s) => (
                <SelectItem key={s} value={s}>
                  {toc.pages[s]?.title ?? s}
                </SelectItem>
              ))}
            </SelectGroup>
          ) : (
            <SelectItem key={n.slug} value={n.slug!}>
              {n.title}
            </SelectItem>
          ),
        )}
      </SelectContent>
    </Select>
  );
}

function DocsContent({ sub }: { sub: string | null }) {
  const locale = useLocale();
  const slug = slugOf(sub);
  const heading = routeQuery().get("h");
  const [toc, setToc] = useState<DocsToc | null>(null);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let live = true;
    setFailed(false);
    loadToc(locale).then(
      (v) => live && setToc(v),
      () => live && setFailed(true),
    );
    return () => {
      live = false;
    };
  }, [locale]);

  const load = useMemo(() => pageLoader(slug, locale), [slug, locale]);
  useEffect(() => {
    setFailed(false);
    if (!load || slug === "api/reference") return;
    let live = true;
    load().then(
      (m) => live && setLoaded({ slug, locale, Page: m.default }),
      () => live && setFailed(true),
    );
    return () => {
      live = false;
    };
  }, [load, slug, locale]);

  const ready = loaded && loaded.slug === slug && loaded.locale === locale;
  useEffect(() => {
    if (!ready) return;
    if (!heading) {
      document.querySelector('[data-testid="page-scroll"]')?.scrollTo(0, 0);
      return;
    }
    // Code blocks and fonts settle after the first paint and move what is above the heading,
    // so the jump is repeated briefly instead of trusting the first layout.
    const go = () =>
      document.getElementById(heading)?.scrollIntoView({ block: "start" });
    go();
    const timers = [150, 500, 1200].map((ms) => setTimeout(go, ms));
    return () => timers.forEach(clearTimeout);
  }, [ready, slug, heading]);
  useEffect(() => {
    setDocTitle(
      slug === "api/reference"
        ? t("docs.reference")
        : (toc?.pages[slug]?.title ?? null),
    );
    return () => setDocTitle(null);
  }, [toc, slug, locale]);
  const components = useMemo(() => mdxComponents(slug), [slug]);
  const info = toc?.pages[slug];
  const order = toc ? leaves(toc.tree) : [];
  const at = order.indexOf(slug);
  const prev = at > 0 ? order[at - 1] : null;
  const next = at >= 0 && at < order.length - 1 ? order[at + 1] : null;

  if (toc && !info && slug !== "api/reference")
    return (
      <DashboardPage data-testid="docs">
        <div role="status" className="max-w-xl space-y-3 py-10">
          <h1 className="text-2xl font-semibold tracking-tight">
            {t("docs.notFound.title")}
          </h1>
          <p className="text-sm text-muted-foreground">
            {t("docs.notFound.body")}
          </p>
          <a
            href={docHref("index")}
            className="text-sm font-medium underline underline-offset-4"
          >
            {t("docs.notFound.back")}
          </a>
        </div>
      </DashboardPage>
    );

  const toc2 = (info?.headings ?? []).filter((h) => h.level <= 3);
  return (
    <DashboardPage width="7xl" data-testid="docs">
      <AnchorProvider
        key={`${slug}-${locale}-${ready}`}
        toc={toc2.map((h) => ({
          title: h.text,
          url: `#${h.id}`,
          depth: h.level,
        }))}
      >
        <div className="grid gap-8 lg:grid-cols-[14rem_minmax(0,1fr)] xl:grid-cols-[14rem_minmax(0,1fr)_13rem]">
          <nav
            aria-label={t("docs.nav.label")}
            data-testid="docs-nav"
            className="lg:sticky lg:top-2 lg:max-h-[calc(100dvh-9rem)] lg:self-start lg:overflow-y-auto"
          >
            {toc && (
              <>
                <div className="lg:hidden">
                  <PageJump toc={toc} slug={slug} />
                </div>
                <a
                  href={docHref("api/reference")}
                  className="mb-4 block rounded-lg border border-border px-3 py-2 text-sm font-medium"
                >
                  {t("docs.reference")}
                </a>
                <div className="hidden lg:block">
                  <Tree node={toc.tree} slug={slug} />
                </div>
              </>
            )}
          </nav>
          <article
            className="prose min-w-0 max-w-3xl"
            data-testid="docs-article"
          >
            <nav
              className="not-prose mb-4 flex flex-wrap gap-2 text-xs text-muted-foreground"
              aria-label={t("docs.breadcrumb")}
            >
              <a href={docHref("index")}>{t("docs.link.lead")}</a>
              {slug !== "index" && (
                <>
                  {info?.breadcrumbs?.map((item, index) => (
                    <span key={index} className="contents">
                      <span>/</span>
                      {item.url ? (
                        <a href={resolveDocLink(item.url, slug) ?? item.url}>
                          {item.title}
                        </a>
                      ) : (
                        <span>{item.title}</span>
                      )}
                    </span>
                  ))}
                  <span>/</span>
                  <span>{info?.title ?? t("docs.reference")}</span>
                </>
              )}
            </nav>
            {slug === "api/reference" ? (
              <Suspense fallback={<Spinner />}>
                <ApiReference />
              </Suspense>
            ) : failed ? (
              <div role="alert" className="space-y-2 py-6">
                <h1 className="text-xl font-semibold">
                  {t("docs.error.title")}
                </h1>
                <p className="text-sm text-muted-foreground">
                  {t("docs.error.body")}
                </p>
              </div>
            ) : (
              <>
                {info && (
                  <header className="mb-6 space-y-2">
                    <h1 className="text-2xl font-semibold tracking-tight">
                      {info.title}
                    </h1>
                    {info.description && (
                      <p className="text-sm leading-7 text-muted-foreground">
                        {info.description}
                      </p>
                    )}
                  </header>
                )}
                {info && <PageActions slug={slug} />}
                {ready ? (
                  <Suspense fallback={null}>
                    <loaded.Page components={components} />
                  </Suspense>
                ) : (
                  <div
                    role="status"
                    className="flex items-center gap-2 py-6 text-sm text-muted-foreground"
                  >
                    <Spinner className="size-4" />
                    {t("docs.loading")}
                  </div>
                )}
                {info?.lastUpdated && (
                  <p className="mt-8 text-xs text-muted-foreground">
                    {t("docs.updated")}{" "}
                    <time dateTime={info.lastUpdated}>
                      {new Date(info.lastUpdated).toLocaleDateString(locale)}
                    </time>
                  </p>
                )}
                {(prev || next) && toc && (
                  <div className="mt-12 grid gap-3 border-t border-border pt-6 sm:grid-cols-2">
                    {prev ? (
                      <a
                        href={docHref(prev)}
                        className="flex items-center gap-2 rounded-xl border border-border bg-(--bg-card) p-3 text-sm hover:bg-(--bg-elevated)"
                      >
                        <ChevronLeft size={16} aria-hidden="true" />
                        <span className="min-w-0">
                          <span className="block text-xs text-muted-foreground">
                            {t("docs.prev")}
                          </span>
                          <span className="block truncate font-medium">
                            {toc.pages[prev]?.title}
                          </span>
                        </span>
                      </a>
                    ) : (
                      <span />
                    )}
                    {next && (
                      <a
                        href={docHref(next)}
                        className="flex items-center justify-end gap-2 rounded-xl border border-border bg-(--bg-card) p-3 text-right text-sm hover:bg-(--bg-elevated)"
                      >
                        <span className="min-w-0">
                          <span className="block text-xs text-muted-foreground">
                            {t("docs.next")}
                          </span>
                          <span className="block truncate font-medium">
                            {toc.pages[next]?.title}
                          </span>
                        </span>
                        <ChevronRight size={16} aria-hidden="true" />
                      </a>
                    )}
                  </div>
                )}
              </>
            )}
          </article>
          {toc2.length > 1 && (
            <aside
              aria-label={t("docs.toc.title")}
              className="hidden xl:sticky xl:top-2 xl:block xl:max-h-[calc(100dvh-9rem)] xl:self-start xl:overflow-y-auto"
            >
              <p className="mb-2 text-xs font-medium text-muted-foreground">
                {t("docs.toc.title")}
              </p>
              <ul className="space-y-1 border-l border-border">
                {toc2.map((h) => (
                  <li key={h.id}>
                    <TOCItem
                      href={`#${h.id}`}
                      onClick={(event) => {
                        event.preventDefault();
                        location.hash = docHref(slug, h.id).slice(1);
                      }}
                      className={`block py-0.5 text-sm text-muted-foreground transition-colors duration-(--dur-fast) ease-out hover:text-foreground data-[active=true]:font-medium data-[active=true]:text-foreground motion-reduce:transition-none ${h.level === 3 ? "pl-6" : "pl-3"}`}
                    >
                      {h.text}
                    </TOCItem>
                  </li>
                ))}
              </ul>
            </aside>
          )}
        </div>
      </AnchorProvider>
    </DashboardPage>
  );
}

export default function Docs(props: { sub: string | null }) {
  return (
    <DocsProvider>
      <DocsContent {...props} />
    </DocsProvider>
  );
}
