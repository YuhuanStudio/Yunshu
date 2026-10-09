import {
  Children,
  isValidElement,
  type ComponentProps,
  type ReactNode,
} from "react";
import {
  Alert,
  InlineCode,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@yuhuanowo/yunui";
import { ChevronDown, Link2 } from "lucide-react";
import { CodeBlock } from "../LazyCodeBlock";
import { t } from "../i18n/index.ts";
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

function Callout({ type, children }: { type?: string; children?: ReactNode }) {
  return (
    <Alert
      variant={type === "warn" || type === "warning" ? "warning" : "info"}
      className="my-4 text-sm [&_p]:my-1"
    >
      {children}
    </Alert>
  );
}

function DocTabs({
  items,
  children,
}: {
  items: string[];
  children?: ReactNode;
}) {
  return (
    <Tabs defaultValue={items[0]} className="my-4">
      <TabsList className="max-w-full overflow-x-auto">
        {items.map((i) => (
          <TabsTrigger key={i} value={i}>
            {i}
          </TabsTrigger>
        ))}
      </TabsList>
      {children}
    </Tabs>
  );
}
const Tab = ({ value, children }: { value: string; children?: ReactNode }) => (
  <TabsContent value={value}>{children}</TabsContent>
);

function Accordions({ children }: { children?: ReactNode }) {
  return <div className="my-4 space-y-2">{children}</div>;
}
function Accordion({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <details className="group rounded-xl border border-border bg-(--bg-card) px-4">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 py-3 text-sm font-medium [&::-webkit-details-marker]:hidden">
        {title}
        <ChevronDown
          size={16}
          aria-hidden="true"
          className="shrink-0 text-muted-foreground transition-transform group-open:rotate-180"
        />
      </summary>
      <div className="pb-2 text-sm [&>:first-child]:mt-0">{children}</div>
    </details>
  );
}

function Steps({ children }: { children?: ReactNode }) {
  return <ol className="docs-steps my-4 space-y-4">{children}</ol>;
}
const Step = ({ children }: { children?: ReactNode }) => (
  <li className="docs-step relative pl-10">{children}</li>
);

function Cards({ children }: { children?: ReactNode }) {
  return <div className="my-4 grid gap-3 sm:grid-cols-2">{children}</div>;
}
function Card({
  title,
  href,
  children,
}: {
  title: string;
  href?: string;
  children?: ReactNode;
}) {
  const link = href ? (resolveDocLink(href, "index") ?? href) : null;
  const body = (
    <>
      <span className="block text-sm font-medium">{title}</span>
      {children ? (
        <span className="mt-1 block text-sm text-muted-foreground">
          {children}
        </span>
      ) : null}
    </>
  );
  const cls =
    "block rounded-xl border border-border bg-(--bg-card) p-4 transition-colors hover:bg-(--bg-elevated)";
  return link ? (
    <a href={link} className={cls}>
      {body}
    </a>
  ) : (
    <div className={cls}>{body}</div>
  );
}

function heading(level: 1 | 2 | 3 | 4, slug: string) {
  const Tag = `h${level}` as const;
  const size = {
    1: "mt-0 mb-3 text-2xl font-semibold tracking-tight",
    2: "mt-10 mb-3 text-xl font-semibold tracking-tight",
    3: "mt-7 mb-2 text-base font-semibold",
    4: "mt-5 mb-2 text-sm font-semibold",
  }[level];
  return function Heading({ id, children }: ComponentProps<"h2">) {
    return (
      <Tag id={id} className={`group scroll-mt-4 ${size}`}>
        {children}
        {id && (
          <a
            href={docHref(slug, id)}
            aria-label={t("docs.copyLink")}
            className="ml-2 inline-block align-middle text-muted-foreground opacity-0 transition-opacity focus-visible:opacity-100 group-hover:opacity-100"
          >
            <Link2 size={14} />
          </a>
        )}
      </Tag>
    );
  };
}

/** The element map MDX renders with: console styling for markdown and the docs' own blocks. */
export function mdxComponents(slug: string) {
  return {
    h1: heading(1, slug),
    h2: heading(2, slug),
    h3: heading(3, slug),
    h4: heading(4, slug),
    p: (p: ComponentProps<"p">) => (
      <p className="my-3 text-sm leading-7 text-foreground" {...p} />
    ),
    ul: (p: ComponentProps<"ul">) => (
      <ul
        className="my-3 list-disc space-y-1.5 pl-6 text-sm leading-7"
        {...p}
      />
    ),
    ol: (p: ComponentProps<"ol">) => (
      <ol
        className="my-3 list-decimal space-y-1.5 pl-6 text-sm leading-7"
        {...p}
      />
    ),
    li: (p: ComponentProps<"li">) => <li className="pl-1" {...p} />,
    strong: (p: ComponentProps<"strong">) => (
      <strong className="font-semibold" {...p} />
    ),
    hr: () => <hr className="my-8 border-border" />,
    blockquote: (p: ComponentProps<"blockquote">) => (
      <blockquote
        className="my-4 border-l-2 border-border pl-4 text-sm text-muted-foreground"
        {...p}
      />
    ),
    a: ({ href = "", children, ...rest }: ComponentProps<"a">) => {
      const link = resolveDocLink(href, slug);
      return link ? (
        <a
          href={link}
          className="font-medium underline underline-offset-4"
          {...rest}
        >
          {children}
        </a>
      ) : (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium underline underline-offset-4"
          {...rest}
        >
          {children}
        </a>
      );
    },
    code: (p: ComponentProps<"code">) => <InlineCode {...p} />,
    pre: ({ children }: ComponentProps<"pre">) => {
      const child = Children.toArray(children)[0];
      if (isValidElement<{ className?: string; children?: ReactNode }>(child)) {
        const lang = /language-([\w-]+)/.exec(child.props.className ?? "")?.[1];
        const code = String(child.props.children ?? "").replace(/\n$/, "");
        return (
          <div className="my-4 min-w-0">
            <CodeBlock language={lang ?? "text"}>{code}</CodeBlock>
          </div>
        );
      }
      return <pre className="my-4 overflow-x-auto text-xs">{children}</pre>;
    },
    table: (p: ComponentProps<"table">) => (
      <Table containerClassName="my-4 rounded-xl border border-border" {...p} />
    ),
    thead: (p: ComponentProps<"thead">) => <TableHeader {...p} />,
    tbody: (p: ComponentProps<"tbody">) => <TableBody {...p} />,
    tr: (p: ComponentProps<"tr">) => <TableRow {...p} />,
    th: ({ align, ...p }: ComponentProps<"th">) => (
      <TableHead
        align={align === "justify" || align === "char" ? undefined : align}
        {...p}
      />
    ),
    td: (p: ComponentProps<"td">) => (
      <TableCell className="min-w-24 align-top [&_code]:break-all" {...p} />
    ),
    Endpoint,
    Support,
    NotServed,
    Callout,
    Tabs: DocTabs,
    Tab,
    Accordions,
    Accordion,
    Steps,
    Step,
    Cards,
    Card,
  };
}
