import { useMemo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";
import type { Meta } from "./api";

/** Map a markdown link target to an in-app route (or null: leave as plain text). */
export function resolveLink(href: string, docId: string | null, meta: Meta | null): { href: string; external: boolean } | null {
  if (!href) return null;
  if (/^(https?:|mailto:)/i.test(href)) return { href, external: true };
  if (href.startsWith("#")) return null; // in-page anchors are handled by the browser via rehype-slug ids
  const [pathPart, hash = ""] = href.split("#");
  const decoded = (() => {
    try {
      return decodeURIComponent(pathPart);
    } catch {
      return pathPart;
    }
  })();
  let id: string | null = null;
  if (decoded.startsWith("/")) {
    if (meta && decoded.startsWith(meta.researchRoot + "/")) id = "r/" + decoded.slice(meta.researchRoot.length + 1);
    else if (meta && decoded.startsWith(meta.codexRoot + "/")) id = "c/" + decoded.slice(meta.codexRoot.length + 1);
  } else if (docId) {
    const [kind, ...rest] = docId.split("/");
    const parts = rest.slice(0, -1);
    for (const seg of decoded.split("/")) {
      if (seg === "." || seg === "") continue;
      if (seg === "..") {
        if (!parts.length) return null;
        parts.pop();
      } else parts.push(seg);
    }
    id = `${kind}/${parts.join("/")}`;
  }
  if (!id) return null;
  if (!/\.(md|txt|json|log|csv|jsonl)$/i.test(id)) return null;
  return { href: `#/docs/${id.split("/").map(encodeURIComponent).join("/")}${hash ? "?h=" + encodeURIComponent(hash) : ""}`, external: false };
}

export function Markdown({ text, docId, meta }: { text: string; docId: string | null; meta: Meta | null }) {
  const components = useMemo<Components>(
    () => ({
      a({ href, children }) {
        const r = resolveLink(href ?? "", docId, meta);
        if (href?.startsWith("#"))
          return (
            <a
              href={href}
              onClick={(e) => {
                e.preventDefault(); // the app routes on location.hash, so anchors scroll manually
                document.getElementById(decodeURIComponent(href.slice(1)))?.scrollIntoView({ behavior: "smooth" });
              }}
            >
              {children}
            </a>
          );
        if (!r) return <code title={href}>{children}</code>;
        return r.external ? (
          <a href={r.href} target="_blank" rel="noreferrer noopener">
            {children}
          </a>
        ) : (
          <a href={r.href}>{children}</a>
        );
      },
      table: ({ children }) => (
        <div className="rs-table" tabIndex={0} role="region" aria-label="表格（可左右捲動）">
          <table>{children}</table>
        </div>
      ),
      img: () => null,
    }),
    [docId, meta],
  );
  return (
    <div className="rs-md">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSlug]} components={components} skipHtml>
        {text}
      </ReactMarkdown>
    </div>
  );
}
