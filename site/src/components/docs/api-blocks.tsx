import type { ReactNode } from "react";

const METHOD_TONE: Record<string, string> = {
  GET: "text-emerald-700 dark:text-emerald-300 bg-emerald-500/10 border-emerald-500/30",
  POST: "text-sky-700 dark:text-sky-300 bg-sky-500/10 border-sky-500/30",
  DELETE: "text-rose-700 dark:text-rose-300 bg-rose-500/10 border-rose-500/30",
  PUT: "text-amber-700 dark:text-amber-300 bg-amber-500/10 border-amber-500/30",
  WS: "text-violet-700 dark:text-violet-300 bg-violet-500/10 border-violet-500/30",
};

/** A single endpoint line: `<Endpoint method="POST" path="/v1/chat/completions" />`. */
export function Endpoint({ method, path, children }: { method: string; path: string; children?: ReactNode }) {
  const tone = METHOD_TONE[method] ?? METHOD_TONE.GET;
  return (
    <div className="not-prose my-3 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-fd-border bg-fd-card px-3 py-2">
      <span className={`rounded-md border px-2 py-0.5 font-mono text-xs font-semibold ${tone}`}>{method}</span>
      <code className="break-all font-mono text-sm text-fd-foreground">{path}</code>
      {children ? <span className="text-sm text-fd-muted-foreground">{children}</span> : null}
    </div>
  );
}

const STATUS_TONE: Record<string, string> = {
  supported: "text-emerald-700 dark:text-emerald-300 bg-emerald-500/10 border-emerald-500/30",
  partial: "text-amber-700 dark:text-amber-300 bg-amber-500/10 border-amber-500/30",
  unsupported: "text-rose-700 dark:text-rose-300 bg-rose-500/10 border-rose-500/30",
  planned: "text-slate-700 dark:text-slate-300 bg-slate-500/10 border-slate-500/30",
  extension: "text-violet-700 dark:text-violet-300 bg-violet-500/10 border-violet-500/30",
};

/** Inline status badge: `<Support status="partial">accepted, ignored</Support>`. */
export function Support({ status, children }: { status: keyof typeof STATUS_TONE; children?: ReactNode }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-px align-middle text-xs font-medium ${STATUS_TONE[status] ?? STATUS_TONE.planned}`}
    >
      {children}
    </span>
  );
}

/** Wraps a route that the server does NOT serve, so the route test skips it. */
export function NotServed({ children }: { children: ReactNode }) {
  return <span className="line-through decoration-fd-muted-foreground/60">{children}</span>;
}
