import { Suspense, lazy, useEffect, useRef, useState } from "react";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
  Button,
  CommandPalette,
  IconButton,
  Kbd,
  Spinner,
  Toaster,
  StatusIndicator,
  useCommandPaletteShortcut,
  type CommandPaletteItem,
} from "@yuhuanowo/yunui";
import { Banner, Sidebar } from "@yuhuanowo/yunui/patterns";
import { YunUIProvider } from "@yuhuanowo/yunui/adapters";
import {
  Activity,
  Box,
  Code2,
  Gauge,
  Menu,
  MessageSquare,
  Moon,
  Pause,
  Play,
  Search,
  PanelLeftClose,
  PanelLeftOpen,
  Settings as SettingsIcon,
  Stethoscope,
  Sun,
  X,
} from "lucide-react";
import { useEngine } from "./useEngine";
import { LivePill } from "./LivePill";
import { tabTitle } from "./engineView";
import type { Connection } from "./api";
import { FooterStatus } from "./FooterStatus";
import { LanguageSwitch } from "./LanguageSwitch";
import { has, t, tr, useLocale } from "./i18n/index.ts";
import { ConnectionState, fixed, modelLabel, sizeGb } from "./ui";
const Dashboard = lazy(() =>
  import("./Dashboard").then((m) => ({ default: m.Dashboard })),
);
const Models = lazy(() =>
  import("./Models").then((m) => ({ default: m.Models })),
);
const Requests = lazy(() =>
  import("./Requests").then((m) => ({ default: m.Requests })),
);
const Settings = lazy(() =>
  import("./Settings").then((m) => ({ default: m.Settings })),
);
const ApiView = lazy(() =>
  import("./ApiAccess").then((m) => ({ default: m.ApiView })),
);
import { operationResult } from "./operation-result";
const Diagnostics = lazy(() =>
  import("./Diagnostics").then((m) => ({ default: m.Diagnostics })),
);
const Playground = lazy(() =>
  import("./Playground").then((m) => ({ default: m.Playground })),
);
const PAGES = [
  "overview",
  "diagnostics",
  "models",
  "requests",
  "api",
  "settings",
  "playground",
] as const;
const pageTitle = (page: string) => tr(`shell.page.${page}`);
function route() {
  const [p = "", ...rest] = location.hash.replace(/^#\/?/, "").split("/");
  const page = (PAGES as readonly string[]).includes(p) ? p : "overview";
  let sub: string | null = null;
  try {
    sub =
      page === "models" && rest.length
        ? decodeURIComponent(rest.join("/"))
        : null;
  } catch {
    sub = null;
  }
  return { page, sub };
}
function stored(key: string, fallback: string) {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}
function base() {
  const value = stored("yunshu.console.url", location.origin);
  try {
    const u = new URL(value);
    if (
      ["http:", "https:"].includes(u.protocol) &&
      !u.username &&
      !u.password &&
      !u.search &&
      !u.hash
    )
      return value;
  } catch {}
  return location.origin;
}
/** YunUI asks for `<namespace>.<key>`; its own strings live in the `yunui` dictionary. */
const translators = new Map<string, ReturnType<typeof makeTranslator>>();
function makeTranslator(namespace: string | undefined) {
  return (key: string, values?: Record<string, unknown>) => {
    const full = `yunui.${namespace ? namespace + "." : ""}${key}`;
    // Unknown keys come back as the key itself: YunUI then uses its own English fallback.
    return has(full)
      ? // i18n-keys: yunui.
        tr(full, values as Record<string, string | number>)
      : key;
  };
}
// One stable function per namespace and locale: YunUI puts the translator in effect deps.
const adapters = {
  useT: (namespace?: string) => {
    const locale = useLocale();
    const id = `${locale}:${namespace ?? ""}`;
    let fn = translators.get(id);
    if (!fn) {
      fn = makeTranslator(namespace);
      translators.set(id, fn);
    }
    return fn;
  },
};
export default function App() {
  const locale = useLocale();
  const [{ page, sub }, setRoute] = useState(route),
    [menu, setMenu] = useState(false),
    [collapsed, setCollapsed] = useState(() => {
      try {
        return localStorage.getItem("yunshu.console.sidebar") === "collapsed";
      } catch {
        return false;
      }
    }),
    [dark, setDark] = useState(
      () =>
        stored(
          "yunshu.console.theme",
          matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light",
        ) === "dark",
    ),
    [connection, setConnection] = useState<Connection>(() => ({
      baseUrl: base(),
      token: "",
    })),
    [revision, setRevision] = useState(0),
    [busy, setBusy] = useState<string | null>(null),
    [notice, setNotice] = useState<{ error: boolean; text: string } | null>(
      null,
    ),
    [testModel, setTestModel] = useState("");
  const engine = useEngine(connection);
  const loadedModel = engine.status?.models.find((m) => m.loaded);
  const [palette, setPalette] = useState(false),
    [query, setQuery] = useState("");
  // Focus goes back to whatever opened the palette (button or shortcut).
  const paletteOpener = useRef<HTMLElement | null>(null);
  const openPalette = () => {
    paletteOpener.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    setPalette(true);
  };
  const closePalette = () => {
    setPalette(false);
    setQuery("");
    const opener = paletteOpener.current;
    paletteOpener.current = null;
    if (opener?.isConnected) setTimeout(() => opener.focus(), 0);
    else setTimeout(() => document.getElementById("main-content")?.focus(), 0);
  };
  useCommandPaletteShortcut(openPalette);
  const skipToMain = (event: { preventDefault: () => void }) => {
    event.preventDefault();
    document.getElementById("main-content")?.focus();
  };
  useEffect(() => {
    try {
      localStorage.setItem(
        "yunshu.console.sidebar",
        collapsed ? "collapsed" : "open",
      );
    } catch {
      /* storage may be unavailable */
    }
  }, [collapsed]);
  useEffect(() => {
    document.title = tabTitle(engine.status, pageTitle(page));
  }, [engine.status, page, locale]);
  const commands: CommandPaletteItem[] = [
    ...PAGES.map((key) => ({
      id: "go:" + key,
      title: pageTitle(key),
      group: t("shell.cmd.go"),
      onSelect: () => navigate(key),
    })),
    ...(engine.status?.models ?? []).map((m) => ({
      id: "model:" + m.id,
      title: modelLabel(m.id),
      description: `${m.type} · ${sizeGb(m.size_gb)} · ${m.loaded ? t("shell.cmd.loaded") : t("shell.cmd.notLoaded")}`,
      group: t("shell.cmd.models"),
      onSelect: () => navigate("models"),
    })),
    {
      id: "polling",
      title: engine.polling ? t("shell.cmd.pause") : t("shell.cmd.resume"),
      icon: engine.polling ? <Pause size={14} /> : <Play size={14} />,
      group: t("shell.cmd.actions"),
      onSelect: () => engine.setPolling(!engine.polling),
    },
    {
      id: "theme",
      title: dark ? t("shell.top.themeLight") : t("shell.top.themeDark"),
      icon: dark ? <Sun size={14} /> : <Moon size={14} />,
      group: t("shell.cmd.actions"),
      onSelect: () => setDark((v) => !v),
    },
  ];
  const q = query.trim().toLowerCase();
  const shown = q
    ? commands.filter((c) =>
        `${c.title} ${c.description ?? ""} ${c.id}`.toLowerCase().includes(q),
      )
    : commands;
  useEffect(() => {
    const fn = () => {
      setRoute(route());
      setMenu(false);
    };
    addEventListener("hashchange", fn);
    return () => removeEventListener("hashchange", fn);
  }, []);
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("yunshu.console.theme", dark ? "dark" : "light");
    } catch {}
  }, [dark]);
  function navigate(p: string, id: string | null = null) {
    location.hash = "/" + p + (id ? "/" + encodeURIComponent(id) : "");
    setRoute({ page: p, sub: id });
    setMenu(false);
    setNotice(null);
  }
  async function perform(key: string, action: () => Promise<unknown>) {
    if (busy) return;
    setBusy(key);
    setNotice(null);
    try {
      const result = await action();
      await engine.refresh();
      setNotice(operationResult(key, result));
    } catch (e) {
      setNotice({
        error: true,
        text:
          e instanceof Error && /timed out/i.test(e.message)
            ? t("errors.operation.timeout")
            : e instanceof Error
              ? e.message
              : String(e),
      });
    } finally {
      setBusy(null);
    }
  }
  function save(next: Connection) {
    if (busy) return;
    setConnection(next);
    setRevision((v) => v + 1);
    setNotice(null);
    setTestModel("");
    try {
      localStorage.setItem("yunshu.console.url", next.baseUrl);
    } catch {
      setNotice({
        error: false,
        text: t("errors.operation.saveAddressFailed"),
      });
    }
  }
  return (
    <YunUIProvider adapters={adapters}>
      <Toaster position="bottom-center" offset={56} />
      <div className="relative h-dvh overflow-hidden bg-(--bg-window)">
        <a href="#main-content" className="skip-link" onClick={skipToMain}>
          {t("shell.nav.skip")}
        </a>
        <Sidebar
          appName="Yunshu"
          ariaLabel={t("shell.nav.ariaLabel")}
          currentPath={"/" + page}
          isOpen={menu}
          onClose={() => setMenu(false)}
          closeLabel={t("shell.nav.close")}
          onNavigate={(href) => navigate(href.replace(/^\//, ""))}
          homeHref="/overview"
          collapsed={collapsed}
          onToggleCollapse={() => setCollapsed((v) => !v)}
          loading={engine.phase === "connecting" && !engine.status}
          header={
            <div className="flex items-center gap-2.5 px-4 pb-4 pt-5">
              <div className="flex min-w-0 flex-1 items-center gap-2.5 px-2">
                <CloudMark />
                <span className="flex-1 truncate text-base font-semibold tracking-tight">
                  Yunshu
                  <span className="ml-1.5 text-xs font-normal text-muted-foreground">
                    {t("shell.brand.name")}
                  </span>
                </span>
              </div>
              <IconButton
                className="hidden lg:inline-flex"
                icon={<PanelLeftClose size={17} />}
                label={t("shell.nav.collapse")}
                onClick={() => setCollapsed(true)}
              />
              <IconButton
                className="lg:hidden"
                icon={<X size={17} />}
                label={t("shell.nav.close")}
                onClick={() => setMenu(false)}
              />
            </div>
          }
          sections={[
            {
              title: t("shell.nav.section.monitor"),
              items: [
                {
                  label: t("shell.page.overview"),
                  href: "/overview",
                  icon: Gauge,
                },
                {
                  label: t("shell.page.requests"),
                  href: "/requests",
                  icon: Activity,
                },
                {
                  label: t("shell.page.diagnostics"),
                  href: "/diagnostics",
                  icon: Stethoscope,
                },
              ],
            },
            {
              title: t("shell.nav.section.models"),
              items: [
                { label: t("shell.page.models"), href: "/models", icon: Box },
              ],
            },
            {
              title: t("shell.nav.section.develop"),
              items: [
                {
                  label: t("shell.page.playground"),
                  href: "/playground",
                  icon: MessageSquare,
                },
                { label: t("shell.page.api"), href: "/api", icon: Code2 },
              ],
            },
          ]}
          footer={
            <>
              <Button
                variant="outline"
                className="mb-3 h-auto rounded-[20px] bg-(--bg-card) w-full flex-col items-start gap-0 px-3 py-2.5 text-left font-normal hover:bg-(--bg-elevated)"
                onClick={() => navigate("models")}
              >
                <span className="sr-only">{t("shell.side.openModels")}</span>
                <span className="mb-1 flex items-center gap-2 text-xs text-muted-foreground">
                  <StatusIndicator
                    status={
                      engine.phase !== "online"
                        ? "offline"
                        : loadedModel
                          ? "online"
                          : "neutral"
                    }
                  />
                  {engine.phase !== "online"
                    ? t("shell.side.offline")
                    : loadedModel
                      ? t("shell.side.loaded")
                      : t("shell.side.noModel")}
                </span>
                <span
                  className={`block w-full truncate text-base font-semibold ${engine.phase === "online" ? "" : "text-muted-foreground"}`}
                >
                  {loadedModel
                    ? modelLabel(loadedModel.id)
                    : engine.phase === "online"
                      ? t("shell.side.pick")
                      : t("shell.side.waiting")}
                </span>
                <span className="mt-0.5 block w-full truncate text-xs tabular-nums text-muted-foreground">
                  {t("shell.side.memory", {
                    used: fixed(engine.status?.memory.active_gb),
                    total: fixed(engine.status?.memory.total_gb),
                  })}
                </span>
              </Button>
              <Button
                variant="outline"
                className={`h-auto rounded-[20px] bg-(--bg-card) w-full justify-start gap-3 px-3 py-2.5 text-left font-normal hover:bg-(--bg-elevated) ${page === "settings" ? "bg-(--bg-elevated)" : ""}`}
                aria-current={page === "settings" ? "page" : undefined}
                onClick={() => navigate("settings")}
              >
                <SettingsIcon
                  size={16}
                  className="shrink-0 text-muted-foreground"
                />
                <span className="min-w-0 flex-1">
                  <span className="sr-only">
                    {t("shell.side.openSettings")}
                  </span>
                  <span className="block truncate text-sm font-medium">
                    {(() => {
                      try {
                        return new URL(connection.baseUrl, location.href).host;
                      } catch {
                        return connection.baseUrl;
                      }
                    })()}
                  </span>
                  <span className="block truncate text-xs tabular-nums text-muted-foreground">
                    {engine.status?.version
                      ? `yunshu ${engine.status.version}`
                      : t("shell.side.localFirst")}
                  </span>
                </span>
              </Button>
            </>
          }
        />
        <div
          className={`flex h-dvh min-w-0 flex-col transition-[padding] duration-150 ease-in-out ${collapsed ? "lg:pl-0" : "lg:pl-64"}`}
        >
          <header className="sticky top-0 z-30 flex shrink-0 items-center gap-4 px-4 pt-4 lg:px-6">
            <IconButton
              className="-ml-2 lg:hidden"
              icon={<Menu size={20} />}
              label={t("shell.nav.open")}
              onClick={() => setMenu(true)}
            />
            {/* Reopen button: inert while the sidebar is open so the collapsed
                animation (max-w-0, opacity-0) cannot leave an invisible tab stop. */}
            <Button
              variant="ghost"
              type="button"
              inert={!collapsed || undefined}
              onClick={() => setCollapsed(false)}
              aria-label={t("shell.nav.expand")}
              className={`hidden shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-all duration-200 ease-in-out hover:bg-muted hover:text-foreground lg:flex ${collapsed ? "-ml-2 max-w-12 p-2 opacity-100" : "pointer-events-none -ml-4 max-w-0 overflow-hidden p-0 opacity-0"}`}
            >
              <PanelLeftOpen size={18} className="shrink-0" />
            </Button>
            <Breadcrumb
              aria-label={t("shell.nav.breadcrumb")}
              className="card w-fit min-w-0 whitespace-nowrap px-3 py-2"
            >
              <BreadcrumbList className="flex-nowrap gap-2 overflow-hidden sm:gap-2">
                <BreadcrumbItem className="shrink-0">
                  <BreadcrumbLink href="#/overview">
                    {t("shell.brand.name")}
                  </BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator />
                <BreadcrumbItem className="min-w-0">
                  {sub ? (
                    <BreadcrumbLink href="#/models">
                      {pageTitle(page)}
                    </BreadcrumbLink>
                  ) : (
                    <BreadcrumbPage className="truncate">
                      {pageTitle(page)}
                    </BreadcrumbPage>
                  )}
                </BreadcrumbItem>
                {sub && (
                  <>
                    <BreadcrumbSeparator />
                    <BreadcrumbItem className="min-w-0">
                      <BreadcrumbPage className="truncate">
                        {modelLabel(sub)}
                      </BreadcrumbPage>
                    </BreadcrumbItem>
                  </>
                )}
              </BreadcrumbList>
            </Breadcrumb>
            <div className="ml-auto flex shrink-0 items-center gap-1.5">
              <LivePill phase={engine.phase} status={engine.status} />
              <Button
                variant="ghost"
                type="button"
                onClick={openPalette}
                className="card hidden h-8 items-center gap-1.5 rounded-full px-3 py-0 text-xs text-muted-foreground transition-colors hover:text-foreground sm:inline-flex"
              >
                <Search size={13} />
                {t("shell.top.search")}
                <Kbd>⌘K</Kbd>
              </Button>
              {/* YunUI ThemeToggle is next-themes backed; the console owns its
                  theme state (Settings shares it), so keep a pill IconButton. */}
              <LanguageSwitch variant="pill" className="hidden sm:block" />
              <IconButton
                className="card size-8 rounded-full"
                icon={dark ? <Sun size={16} /> : <Moon size={16} />}
                label={
                  dark ? t("shell.top.themeLight") : t("shell.top.themeDark")
                }
                onClick={() => setDark((v) => !v)}
              />
            </div>
          </header>
          <main
            id="main-content"
            tabIndex={-1}
            className="flex min-h-0 flex-1 flex-col outline-none"
          >
            {(busy || notice) && (
              <div className="mx-auto w-full max-w-7xl shrink-0 space-y-2 px-4 pt-4 lg:px-6">
                {busy && (
                  <div role="status">
                    <Banner
                      tone="neutral"
                      icon={<Spinner size="sm" />}
                      title={t("shell.busy.waiting", {
                        action: busyAction(busy),
                      })}
                    />
                  </div>
                )}
                {notice && (
                  <div role={notice.error ? "alert" : "status"}>
                    <Banner
                      tone={notice.error ? "critical" : "info"}
                      title={notice.text}
                      dismissible
                      dismissLabel={t("shell.busy.dismiss")}
                      onDismiss={() => setNotice(null)}
                    />
                  </div>
                )}
              </div>
            )}
            <div key={revision} className="flex min-h-0 flex-1 flex-col">
              {
                <div
                  className={
                    engine.phase === "online"
                      ? "hidden"
                      : "mx-auto w-full max-w-7xl shrink-0 px-4 pt-4 lg:px-6"
                  }
                >
                  <ConnectionState
                    engine={engine}
                    configure={() => navigate("settings")}
                  />
                </div>
              }
              <Suspense fallback={<PageFallback />}>
                {page === "playground" ? (
                  <Playground
                    connection={connection}
                    engine={engine}
                    initialModel={testModel}
                  />
                ) : (
                  <div className="relative min-h-0 flex-1 overflow-y-auto p-4 pb-6 lg:p-6">
                    <div className="mx-auto w-full max-w-7xl">
                      {page === "diagnostics" && (
                        <Diagnostics connection={connection} engine={engine} />
                      )}
                      {page === "overview" && (
                        <Dashboard engine={engine} navigate={navigate} />
                      )}
                      {page === "models" && (
                        <Models
                          engine={engine}
                          connection={connection}
                          perform={perform}
                          busy={busy}
                          selected={sub}
                          open={(id) => navigate("models", id)}
                          test={(id) => {
                            setTestModel(id);
                            navigate("playground");
                          }}
                        />
                      )}
                      {page === "requests" && (
                        <Requests
                          engine={engine}
                          connection={connection}
                          perform={perform}
                          busy={busy}
                        />
                      )}
                      {page === "settings" && (
                        <Settings
                          connection={connection}
                          save={save}
                          dark={dark}
                          setDark={setDark}
                          disabled={!!busy}
                          engine={engine}
                          perform={perform}
                        />
                      )}
                      {page === "api" && (
                        <ApiView connection={connection} engine={engine} />
                      )}
                    </div>
                  </div>
                )}
              </Suspense>
            </div>
          </main>
          <footer className="shrink-0 border-t border-border/60 bg-(--bg-window)">
            <FooterStatus engine={engine} connection={connection} />
          </footer>
        </div>
        <CommandPalette
          open={palette}
          onClose={closePalette}
          query={query}
          onQueryChange={setQuery}
          items={shown}
          empty={
            <p className="p-4 text-sm text-muted-foreground">
              {t("shell.cmd.empty")}
            </p>
          }
        />
      </div>
    </YunUIProvider>
  );
}

const BUSY_KINDS = [
  "load",
  "unload",
  "warmup",
  "pull",
  "copy",
  "delete",
] as const;
/** What the engine is busy doing, as a sentence start, from the action key (`load:<id>`). */
function busyAction(key: string) {
  const kind = BUSY_KINDS.find((k) => key.startsWith(`${k}:`)) ?? "cancel";
  return tr(`shell.busy.${kind}`);
}

/** Holds the page area while a route chunk loads; same box as a page, no spinner. */
function PageFallback() {
  return (
    <div
      className="min-h-0 flex-1"
      aria-busy="true"
      data-testid="page-loading"
    />
  );
}

/** Brand mark: a pivot (樞) with cloud arcs turning around it. Host content. */
function CloudMark() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      className="text-foreground"
    >
      <path
        d="M4.5 12a7.5 7.5 0 0 1 12.8-5.3"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M19.5 12a7.5 7.5 0 0 1-12.8 5.3"
        stroke="currentColor"
        strokeOpacity=".5"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle cx="12" cy="12" r="2.6" fill="currentColor" />
    </svg>
  );
}
