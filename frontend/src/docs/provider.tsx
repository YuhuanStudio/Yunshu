import { type ComponentProps, type ReactNode } from "react";
import { FrameworkProvider } from "fumadocs-core/framework";
import { RootProvider } from "fumadocs-ui/provider/base";
import { useLocale } from "../i18n";
import { resolveDocLink } from "./links";

// The console owns hash navigation and theme; adapt the framework boundary.
const framework = {
  usePathname: () => location.hash.slice(1).split("?")[0],
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
    <a href={resolveDocLink(href, "index") ?? href} {...props} />
  ),
};
export function DocsProvider({ children }: { children: ReactNode }) {
  const locale = useLocale();
  return (
    <FrameworkProvider {...framework}>
      <RootProvider
        search={{ enabled: false }}
        theme={{ enabled: false }}
        i18n={{ locale, defaultLanguage: "en" }}
      >
        {children}
      </RootProvider>
    </FrameworkProvider>
  );
}
