import Link from "next/link";
import type { ComponentProps } from "react";
import type { Lang } from "@/lib/i18n";

/** MDX `a` override: absolute `/docs/...` links get the page's language prefix. */
export function createLink(lang: Lang) {
  return function LocalizedLink({ href, ...props }: ComponentProps<"a">) {
    if (typeof href === "string" && href.startsWith("/docs")) {
      return <Link href={`/${lang}${href}`} {...(props as object)} />;
    }
    if (typeof href === "string" && href.startsWith("/") && !href.startsWith("//")) {
      return <Link href={href} {...(props as object)} />;
    }
    const external = typeof href === "string" && /^https?:/.test(href);
    return <a href={href} {...props} {...(external ? { target: "_blank", rel: "noreferrer noopener" } : {})} />;
  };
}
