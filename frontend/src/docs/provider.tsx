import {
  type ComponentProps,
  type ReactNode,
  useSyncExternalStore,
} from "react";
import { FrameworkProvider } from "fumadocs-core/framework";
import { RootProvider } from "fumadocs-ui/provider/base";
import { t, useLocale } from "../i18n";
import { resolveDocLink } from "./links";

const subscribe = (cb: () => void) => {
  addEventListener("hashchange", cb);
  return () => removeEventListener("hashchange", cb);
};
const pathname = () => location.hash.slice(1).split("?")[0] || "/";

// The console owns hash navigation and theme; adapt the framework boundary.
const framework = {
  usePathname: () => useSyncExternalStore(subscribe, pathname, pathname),
  useParams: () => ({}),
  useRouter: () => ({
    push: (url: string) => {
      location.hash = url;
    },
    refresh: () => location.reload(),
  }),
  Link: ({
    href = "",
    prefetch: _prefetch,
    ...props
  }: ComponentProps<"a"> & { prefetch?: boolean }) => (
    <a
      href={resolveDocLink(href, "index") ?? href}
      aria-current={
        String(props["data-active" as keyof typeof props]) === "true"
          ? "page"
          : undefined
      }
      {...props}
    />
  ),
};

/** The fumadocs UI strings, in the console's language (keys are fumadocs' own). */
const uiStrings = () => ({
  "Open Sidebar(sidebar)(aria-label)": t("docs.fd.openSidebar"),
  "Close Sidebar(sidebar)(aria-label)": t("docs.fd.closeSidebar"),
  "Collapse Sidebar(sidebar)(aria-label)": t("docs.fd.collapse"),
  "On this page(table of contents)": t("docs.toc.title"),
  "No Headings(table of contents)": t("docs.fd.noHeadings"),
  "Previous Page(pagination)": t("docs.prev"),
  "Next Page(pagination)": t("docs.next"),
  "Last updated on(page footer)": t("docs.updated"),
  "Copy Text(code block)(aria-label)": t("docs.fd.copyText"),
  "Copied Text(code block)(aria-label)": t("docs.fd.copiedText"),
});

export function DocsProvider({ children }: { children: ReactNode }) {
  const locale = useLocale();
  return (
    <FrameworkProvider {...framework}>
      <RootProvider
        search={{ enabled: false }}
        theme={{ enabled: false }}
        i18n={{ locale, defaultLanguage: "en", translations: uiStrings() }}
      >
        {children}
      </RootProvider>
    </FrameworkProvider>
  );
}
