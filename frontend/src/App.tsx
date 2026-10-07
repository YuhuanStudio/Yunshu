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
const titles: Record<string, string> = {
  overview: "引擎總覽",
  diagnostics: "引擎診斷",
  models: "模型庫",
  requests: "請求與效能",
  api: "API 接入",
  settings: "設定",
  playground: "推理測試",
};
function route() {
  const [p = "", ...rest] = location.hash.replace(/^#\/?/, "").split("/");
  const page = Object.hasOwn(titles, p) ? p : "overview";
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
const adapters = {
  useT: () => (key: string, values?: Record<string, unknown>) => {
    const strings: Record<string, string> = {
      copy: "複製",
      copied: "已複製",
      "codeBlock.copyFull": "複製完整程式碼",
      "codeBlock.copyError": "無法寫入剪貼簿",
      "codeBlock.showLess": "收合",
      "codeBlock.tabGroup": "程式碼格式",
      "codeBlock.scrollHorizontally": "左右捲動",
      clickToCopy: "點擊複製",
      "message.prompt": "訊息內容",
    };
    if (key === "clickToCopy") return `點擊複製 ${values?.text ?? ""}`.trim();
    if (key === "codeBlock.lineCount") return `${values?.count ?? 0} 行`;
    if (key === "codeBlock.showAll") return `顯示全部 ${values?.count ?? 0} 行`;
    if (key === "codeBlock.scrollRegion")
      return `${values?.language ?? ""} 程式碼`;
    return strings[key] ?? key;
  },
};
export default function App() {
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
    document.title = tabTitle(engine.status, titles[page]);
  }, [engine.status, page]);
  const commands: CommandPaletteItem[] = [
    ...Object.entries(titles).map(([key, title]) => ({
      id: "go:" + key,
      title,
      group: "前往",
      onSelect: () => navigate(key),
    })),
    ...(engine.status?.models ?? []).map((m) => ({
      id: "model:" + m.id,
      title: modelLabel(m.id),
      description: `${m.type} · ${sizeGb(m.size_gb)} · ${m.loaded ? "已載入" : "未載入"}`,
      group: "模型",
      onSelect: () => navigate("models"),
    })),
    {
      id: "polling",
      title: engine.polling ? "暫停狀態更新" : "恢復狀態更新",
      icon: engine.polling ? <Pause size={14} /> : <Play size={14} />,
      group: "操作",
      onSelect: () => engine.setPolling(!engine.polling),
    },
    {
      id: "theme",
      title: dark ? "切換淺色" : "切換深色",
      icon: dark ? <Sun size={14} /> : <Moon size={14} />,
      group: "操作",
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
            ? "等待服務回應逾時；操作可能仍在服務端進行，請重新整理狀態。"
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
        text: "無法儲存服務位址；目前連線仍會使用新設定。",
      });
    }
  }
  return (
    <YunUIProvider adapters={adapters}>
      <Toaster position="bottom-center" offset={56} />
      <div className="relative h-dvh overflow-hidden bg-(--bg-window)">
        <a href="#main-content" className="skip-link" onClick={skipToMain}>
          跳到主要內容
        </a>
        <Sidebar
          appName="Yunshu"
          ariaLabel="控制台導覽"
          currentPath={"/" + page}
          isOpen={menu}
          onClose={() => setMenu(false)}
          closeLabel="關閉導覽"
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
                    雲樞
                  </span>
                </span>
              </div>
              <IconButton
                className="hidden lg:inline-flex"
                icon={<PanelLeftClose size={17} />}
                label="收合導覽"
                onClick={() => setCollapsed(true)}
              />
              <IconButton
                className="lg:hidden"
                icon={<X size={17} />}
                label="關閉導覽"
                onClick={() => setMenu(false)}
              />
            </div>
          }
          sections={[
            {
              title: "監控",
              items: [
                { label: "引擎總覽", href: "/overview", icon: Gauge },
                { label: "請求與效能", href: "/requests", icon: Activity },
                { label: "引擎診斷", href: "/diagnostics", icon: Stethoscope },
              ],
            },
            {
              title: "模型",
              items: [{ label: "模型庫", href: "/models", icon: Box }],
            },
            {
              title: "開發",
              items: [
                { label: "推理測試", href: "/playground", icon: MessageSquare },
                { label: "API 接入", href: "/api", icon: Code2 },
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
                <span className="sr-only">開啟模型庫，</span>
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
                    ? "引擎未連線"
                    : loadedModel
                      ? "已載入"
                      : "沒有已載入模型"}
                </span>
                <span
                  className={`block w-full truncate text-base font-semibold ${engine.phase === "online" ? "" : "text-muted-foreground"}`}
                >
                  {loadedModel
                    ? modelLabel(loadedModel.id)
                    : engine.phase === "online"
                      ? "選擇模型"
                      : "等待引擎"}
                </span>
                <span className="mt-0.5 block w-full truncate text-xs tabular-nums text-muted-foreground">
                  {fixed(engine.status?.memory.active_gb)} /{" "}
                  {fixed(engine.status?.memory.total_gb)} GB
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
                  <span className="sr-only">開啟設定，</span>
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
                      : "本機優先"}
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
              label="開啟導覽"
              onClick={() => setMenu(true)}
            />
            {/* Reopen button: inert while the sidebar is open so the collapsed
                animation (max-w-0, opacity-0) cannot leave an invisible tab stop. */}
            <Button
              variant="ghost"
              type="button"
              inert={!collapsed || undefined}
              onClick={() => setCollapsed(false)}
              aria-label="展開導覽"
              className={`hidden shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-all duration-200 ease-in-out hover:bg-muted hover:text-foreground lg:flex ${collapsed ? "-ml-2 max-w-12 p-2 opacity-100" : "pointer-events-none -ml-4 max-w-0 overflow-hidden p-0 opacity-0"}`}
            >
              <PanelLeftOpen size={18} className="shrink-0" />
            </Button>
            <Breadcrumb
              aria-label="目前位置"
              className="card w-fit min-w-0 whitespace-nowrap px-3 py-2"
            >
              <BreadcrumbList className="flex-nowrap gap-2 overflow-hidden sm:gap-2">
                <BreadcrumbItem className="shrink-0">
                  <BreadcrumbLink href="#/overview">雲樞</BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator />
                <BreadcrumbItem className="min-w-0">
                  {sub ? (
                    <BreadcrumbLink href="#/models">
                      {titles[page]}
                    </BreadcrumbLink>
                  ) : (
                    <BreadcrumbPage className="truncate">
                      {titles[page]}
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
                搜尋
                <Kbd>⌘K</Kbd>
              </Button>
              {/* YunUI ThemeToggle is next-themes backed; the console owns its
                  theme state (Settings shares it), so keep a pill IconButton. */}
              <IconButton
                className="card size-8 rounded-full"
                icon={dark ? <Sun size={16} /> : <Moon size={16} />}
                label={dark ? "切換淺色" : "切換深色"}
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
                      title={`${
                        busy.startsWith("load:")
                          ? "正在載入模型"
                          : busy.startsWith("unload:")
                            ? "正在卸載模型"
                            : busy.startsWith("warmup:")
                              ? "正在預熱模型"
                              : busy.startsWith("pull:")
                                ? "正在下載模型（後端尚未提供進度）"
                                : busy.startsWith("copy:")
                                  ? "正在建立模型別名"
                                  : busy.startsWith("delete:")
                                    ? "正在刪除模型"
                                    : "正在取消請求"
                      }… 等待服務回應。`}
                    />
                  </div>
                )}
                {notice && (
                  <div role={notice.error ? "alert" : "status"}>
                    <Banner
                      tone={notice.error ? "critical" : "info"}
                      title={notice.text}
                      dismissible
                      dismissLabel="關閉操作訊息"
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
          <footer>
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
            <p className="p-4 text-sm text-muted-foreground">沒有符合的項目</p>
          }
        />
      </div>
    </YunUIProvider>
  );
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
