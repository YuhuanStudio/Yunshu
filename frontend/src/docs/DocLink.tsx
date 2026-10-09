import { BookOpen } from "lucide-react";
import { t } from "../i18n/index.ts";
import { docHref } from "./data.ts";

/** A quiet link from a console page to the docs page that explains it. */
export function DocLink({
  slug,
  heading,
  label,
  className,
}: {
  slug: string;
  heading?: string;
  label: string;
  className?: string;
}) {
  return (
    <a
      href={docHref(slug, heading)}
      data-testid="doc-link"
      aria-label={`${t("docs.link.lead")}: ${label}`}
      className={`inline-flex w-fit items-center gap-1.5 text-sm text-muted-foreground underline-offset-4 transition-colors duration-[120ms] ease-out motion-reduce:transition-none hover:text-foreground hover:underline ${className ?? ""}`}
    >
      <BookOpen size={14} aria-hidden="true" />
      {label}
    </a>
  );
}
