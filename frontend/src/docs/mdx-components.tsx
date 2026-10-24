import { Button } from "@yuhuanowo/yunui";
import { Link2 } from "lucide-react";
import { t } from "../i18n";
import { lazy, type ComponentProps, type ReactNode } from "react";
import defaults from "fumadocs-ui/mdx";
import { Callout } from "fumadocs-ui/components/callout";
import { Tabs, Tab } from "fumadocs-ui/components/tabs";
const Accordions = lazy(() =>
  import("fumadocs-ui/components/accordion").then((m) => ({
    default: m.Accordions,
  })),
);
const Accordion = lazy(() =>
  import("fumadocs-ui/components/accordion").then((m) => ({
    default: m.Accordion,
  })),
);
import { Steps, Step } from "fumadocs-ui/components/steps";
import { Cards, Card as FumaCard } from "fumadocs-ui/components/card";
import { TypeTable } from "fumadocs-ui/components/type-table";
import { Files, File, Folder } from "fumadocs-ui/components/files";
import { ImageZoom } from "fumadocs-ui/components/image-zoom";
import { Banner } from "fumadocs-ui/components/banner";
import { Mermaid } from "./Mermaid";
import { docHref, resolveDocLink } from "./data.ts";

const METHOD_TONE: Record<string, string> = {
  GET: "border-success/40 bg-success/10 text-success",
  POST: "border-info/40 bg-info/10 text-info",
  DELETE: "border-error/40 bg-error/10 text-error",
  PUT: "border-warning/40 bg-warning/10 text-warning",
  PATCH: "border-warning/40 bg-warning/10 text-warning",
  WS: "border-accent/40 bg-accent/10 text-accent",
};
const SUPPORT_TONE: Record<string, string> = {
  supported: "border-success/40 bg-success/10 text-success",
  partial: "border-warning/40 bg-warning/10 text-warning",
  unsupported: "border-error/40 bg-error/10 text-error",
  planned: "border-border bg-(--bg-elevated) text-muted-foreground",
  extension: "border-accent/40 bg-accent/10 text-accent",
};

function Endpoint({
  method,
  path,
  children,
}: {
  method: string;
  path: string;
  children?: ReactNode;
}) {
  return (
    <div className="my-3 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl border border-border bg-(--bg-card) px-3 py-2">
      <span
        className={`rounded-md border px-2 py-0.5 font-mono text-xs font-semibold ${METHOD_TONE[method] ?? METHOD_TONE.GET}`}
      >
        {method}
      </span>
      <code className="break-all font-mono text-sm">{path}</code>
      {children ? (
        <span className="text-sm text-muted-foreground">{children}</span>
      ) : null}
    </div>
  );
}

function Support({
  status,
  children,
}: {
  status: string;
  children?: ReactNode;
}) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-px align-middle text-xs font-medium ${SUPPORT_TONE[status] ?? SUPPORT_TONE.planned}`}
    >
      {children}
    </span>
  );
}

const NotServed = ({ children }: { children?: ReactNode }) => (
  <span className="line-through decoration-muted-foreground/60">
    {children}
  </span>
);

/** Preserve authoring names used by existing and parallel content. */
export function mdxComponents(slug: string) {
  const heading =
    (Tag: "h1" | "h2" | "h3" | "h4" | "h5" | "h6") =>
    ({ id, children, ...props }: ComponentProps<"h2">) => (
      <Tag id={id} {...props} className="group/heading scroll-mt-6">
        <a className="no-underline" href={docHref(slug, id)}>
          {children}
        </a>
        {id && (
          <Button
            size="icon"
            variant="ghost"
            className="ml-1 opacity-0 group-hover/heading:opacity-100 focus-visible:opacity-100"
            aria-label={t("docs.copyLink")}
            onClick={() =>
              navigator.clipboard.writeText(
                new URL(docHref(slug, id), location.href).href,
              )
            }
          >
            <Link2 size={14} />
          </Button>
        )}
      </Tag>
    );
  return {
    ...defaults,
    h1: heading("h2"),
    h2: heading("h2"),
    h3: heading("h3"),
    h4: heading("h4"),
    h5: heading("h5"),
    h6: heading("h6"),
    a: ({ href = "", ...props }: ComponentProps<"a">) => (
      <a href={resolveDocLink(href, slug) ?? href} {...props} />
    ),
    Card: ({ href, ...props }: ComponentProps<typeof FumaCard>) => (
      <FumaCard
        href={href ? (resolveDocLink(href, slug) ?? href) : undefined}
        {...props}
      />
    ),
    Endpoint: ({ method, path, children }: ComponentProps<typeof Endpoint>) => (
      <Endpoint method={method} path={path}>
        {children}
        {method !== "WS" && (
          <a
            className="ml-2 text-xs underline"
            href={
              docHref("api/reference") +
              "?operation=" +
              encodeURIComponent(method.toLowerCase() + " " + path)
            }
          >
            {t("docs.reference.link")}
          </a>
        )}
      </Endpoint>
    ),
    Support,
    NotServed,
    Callout,
    Tabs,
    Tab,
    Accordions,
    Accordion,
    Steps,
    Step,
    Cards,
    TypeTable,
    Files,
    File,
    Folder,
    ImageZoom,
    Banner,
    Mermaid,
    img: ImageZoom,
    SSE: ({ children }: { children?: ReactNode }) => (
      <Callout type="info" title={t("docs.sse")}>
        {children}
      </Callout>
    ),
  };
}
