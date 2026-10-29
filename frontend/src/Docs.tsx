import {
  Suspense,
  lazy,
  useEffect,
  useMemo,
  useState,
  type ComponentType,
  type MouseEvent,
} from "react";
import { Button, Spinner } from "@yuhuanowo/yunui";
import { Sparkles } from "lucide-react";
import { BrandMark } from "./BrandMark";
import { DashboardPage } from "@yuhuanowo/yunui/patterns";
import { DocsLayout } from "fumadocs-ui/layouts/docs";
import { useSidebar } from "fumadocs-ui/components/sidebar/base";
import {
  DocsBody,
  DocsDescription,
  DocsPage,
  DocsTitle,
  PageLastUpdate,
} from "fumadocs-ui/layouts/docs/page";
import { DocsProvider } from "./docs/provider";
import { PageActions } from "./docs/PageActions";
import "./docs/fumadocs.css";
import { t, useLocale, type Locale } from "./i18n/index.ts";
import { routeQuery } from "./route";
import { setDocTitle } from "./docs/title-store.ts";
import { docHref, loadToc, pageLoader, slugOf } from "./docs/data.ts";
import { mdxComponents } from "./docs/mdx-components";
import type { DocsToc } from "./docs/types.ts";

const ApiReference = lazy(() => import("./docs/ApiReference"));

type Loaded = {
  slug: string;
  locale: Locale;
  Page: ComponentType<{ components?: import("mdx/types").MDXComponents }>;
};

/** The phone drawer is not a modal dialog in fumadocs; Escape closes it like every other panel here. */
function SidebarEscape() {
  const { open, setOpen } = useSidebar();
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [open, setOpen]);
  return null;
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

  const headings = (info?.headings ?? []).filter((h) => h.level <= 3);
  const reference = slug === "api/reference";
  // The in-page links are "#heading" for the table of contents; in a hash-routed console
  // they must become a route with the heading as its query, or they would replace the route.
  const onTocClick = (event: MouseEvent<HTMLElement>) => {
    const link = (event.target as Element).closest<HTMLAnchorElement>(
      "#nd-toc a, [data-toc-popover] a",
    );
    const href = link?.getAttribute("href");
    if (!href?.startsWith("#") || href.startsWith("#/")) return;
    event.preventDefault();
    location.hash = docHref(slug, href.slice(1)).slice(1);
  };

  return (
    <DashboardPage width="7xl" data-testid="docs">
      {toc ? (
        <DocsLayout
          tree={toc.pageTree}
          nav={{
            title: (
              <span className="inline-flex items-center gap-2">
                <BrandMark size={22} />
                {t("docs.link.lead")}
              </span>
            ),
            url: "/docs",
            children: (
              <Button
                size="sm"
                variant="outline"
                className="ms-3"
                onClick={() => dispatchEvent(new Event("yunshu-docs-ask"))}
              >
                <Sparkles size={14} aria-hidden="true" />
                {t("docs.ask.title")}
              </Button>
            ),
          }}
          links={[{ text: t("docs.reference"), url: "/docs/api/reference" }]}
          themeSwitch={{ enabled: false }}
          searchToggle={{ enabled: false }}
          sidebar={
            {
              collapsible: false,
              defaultOpenLevel: 1,
              "data-testid": "docs-nav",
            } as object
          }
          containerProps={{ onClickCapture: onTocClick }}
        >
          <SidebarEscape />
          <DocsPage
            key={`${slug}-${locale}`}
            data-testid="docs-article"
            full={reference}
            toc={headings.map((h) => ({
              title: h.text,
              url: `#${h.id}`,
              depth: h.level,
            }))}
            tableOfContent={{
              enabled: !reference && headings.length > 1,
              container: {
                role: "complementary",
                "aria-label": t("docs.toc.title"),
              },
            }}
            tableOfContentPopover={{
              enabled: !reference && headings.length > 1,
            }}
            footer={{ enabled: !reference }}
          >
            {reference ? (
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
                  <>
                    <DocsTitle>{info.title}</DocsTitle>
                    {info.description && (
                      <DocsDescription className="mb-0 text-base">
                        {info.description}
                      </DocsDescription>
                    )}
                    <PageActions slug={slug} />
                  </>
                )}
                {ready ? (
                  <DocsBody>
                    <Suspense fallback={null}>
                      <loaded.Page components={components} />
                    </Suspense>
                  </DocsBody>
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
                  <PageLastUpdate date={new Date(info.lastUpdated)} />
                )}
              </>
            )}
          </DocsPage>
        </DocsLayout>
      ) : (
        <article
          data-testid="docs-article"
          className="flex items-center gap-2 py-10 text-sm text-muted-foreground"
          role={failed ? "alert" : "status"}
        >
          {failed ? (
            t("docs.error.body")
          ) : (
            <>
              <Spinner className="size-4" />
              {t("docs.loading")}
            </>
          )}
        </article>
      )}
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
