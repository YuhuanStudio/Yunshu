import { useState } from "react";
import { Button } from "@yuhuanowo/yunui";
import { Copy, Check, ExternalLink } from "lucide-react";
import { t, useLocale } from "../i18n";

export function PageActions({ slug }: { slug: string }) {
  const locale = useLocale();
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");
  const markdown = `${location.origin}/console/docs-assets/${locale}/${slug}.md`;
  const prompt = encodeURIComponent(
    `Read ${markdown} and help me understand this Yunshu documentation. Source page: ${location.href}`,
  );
  return (
    <div className="not-prose my-4 flex flex-wrap items-center gap-2 text-xs">
      <Button
        size="sm"
        variant="outline"
        onClick={async () => {
          setError("");
          try {
            const res = await fetch(markdown);
            if (!res.ok) throw new Error(String(res.status));
            await navigator.clipboard.writeText(await res.text());
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
          } catch {
            setError(t("docs.actions.failed"));
          }
        }}
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
        {t(copied ? "docs.actions.copied" : "docs.actions.copy")}
      </Button>
      <a
        href={`https://chatgpt.com/?q=${prompt}`}
        target="_blank"
        rel="noreferrer"
        className="inline-flex items-center gap-1"
      >
        {t("docs.actions.chatgpt")} <ExternalLink size={12} />
      </a>
      <a
        href={`https://claude.ai/new?q=${prompt}`}
        target="_blank"
        rel="noreferrer"
        className="inline-flex items-center gap-1"
      >
        {t("docs.actions.claude")} <ExternalLink size={12} />
      </a>
      {error && <span role="alert">{error}</span>}
    </div>
  );
}
