/** `#/docs/<slug>[?h=<heading id>]`; the index page is plain `#/docs`. */
export function docHref(slug: string, heading?: string | null): string {
  const path = slug === "index" ? "" : "/" + slug;
  return `#/docs${path}${heading ? "?h=" + encodeURIComponent(heading) : ""}`;
}

/** The slug in a docs route (`#/docs/api/audio`), `index` for `#/docs`. */
export const slugOf = (sub: string | null): string => sub || "index";

/** A link written in an MDX page, as a console route (null: leave it as an external link). */
export function resolveDocLink(href: string, current: string): string | null {
  if (/^(https?:|mailto:)/.test(href)) return null;
  const [path, hash] = href.split("#");
  if (!path) return docHref(current, hash || null);
  const m = /^\/docs\/?(.*?)\/?$/.exec(path);
  if (!m) return null;
  return docHref(m[1] || "index", hash || null);
}
