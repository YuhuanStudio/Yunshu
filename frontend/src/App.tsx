import { useEffect, useState } from "react";
import {
  Button,
  CommandPalette,
  IconButton,
  Kbd,
  StatusIndicator,
  useCommandPaletteShortcut,
  type CommandPaletteItem,
} from "@yuhuanowo/yunui";
import { Sidebar, StatusPill, StatusPillBar } from "@yuhuanowo/yunui/patterns";
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
  Settings as SettingsIcon,
  Stethoscope,
  Sun,
  X,
} from "lucide-react";
import { useEngine } from "./useEngine";
import type { Connection, EngineLastRequest } from "./api";
import { ConnectionState, clock, modelLabel, number } from "./ui";
import { Dashboard } from "./Dashboard";
import { Models } from "./Models";
import { Requests } from "./Requests";
import { Settings, ApiView } from "./Settings";
import { operationResult } from "./operation-result";
import { Diagnostics } from "./Diagnostics";
import { Playground } from "./Playground";
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
  const p = location.hash.replace(/^#\/?/, "");
  return Object.hasOwn(titles, p) ? p : "overview";
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
    };
    if (key === "codeBlock.lineCount") return `${values?.count ?? 0} 行`;
    if (key === "codeBlock.showAll") return `顯示全部 ${values?.count ?? 0} 行`;
    if (key === "codeBlock.scrollRegion")
      return `${values?.language ?? ""} 程式碼`;
    return strings[key] ?? key;
  },
};
export default function App() {
  const [page, setPage] = useState(route),
    [menu, setMenu] = useState(false),
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
  useCommandPaletteShortcut(() => setPalette(true));
  const liveTps =
    engine.status?.throughput.live_decode_tps ??
    (engine.status?.requests.active ? null : undefined);
  useEffect(() => {
    const active = engine.status?.requests.active ?? 0;
    document.title =
      active > 0 && liveTps
        ? `▶ ${number(liveTps)} tok/s · 雲樞`
        : `${titles[page]} · 雲樞 Yunshu`;
  }, [engine.status, liveTps, page]);
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
      description: `${m.type} · ${number(m.size_gb)} GB · ${m.loaded ? "已載入" : "未載入"}`,
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
  const last = engine.status?.last;
  useEffect(() => {
    const fn = () => {
      setPage(route());
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
  function navigate(p: string) {
    location.hash = "/" + p;
    setPage(p);
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
      <div className="h-dvh overflow-hidden bg-background">
        <Sidebar
          appName="Yunshu"
          ariaLabel="控制台導覽"
          currentPath={"/" + page}
          isOpen={menu}
          onClose={() => setMenu(false)}
          closeLabel="關閉導覽"
          onNavigate={(href) => navigate(href.replace(/^\//, ""))}
          header={
            <div className="space-y-4 px-4 pb-3 pt-5">
              <div className="flex items-center gap-2.5 px-2">
                <CloudMark />
                <span className="flex-1 text-[15px] font-semibold tracking-tight">
                  Yunshu
                  <span className="ml-1.5 text-xs font-normal text-muted-foreground">
                    雲樞
                  </span>
                </span>
                <IconButton
                  className="lg:hidden"
                  icon={<X size={17} />}
                  label="關閉導覽"
                  onClick={() => setMenu(false)}
                />
              </div>
              <Button
                variant="secondary"
                className="h-auto w-full flex-col items-start gap-0 px-3 py-2.5 text-left"
                onClick={() => navigate("models")}
              >
                <span className="flex items-center gap-2 text-xs text-muted-foreground">
                  <StatusIndicator
                    status={
                      engine.phase !== "online"
                        ? "offline"
                        : loadedModel
                          ? "online"
                          : "neutral"
                    }
                    pulse={(engine.status?.requests.active ?? 0) > 0}
                  />
                  {engine.phase !== "online"
                    ? "引擎未連線"
                    : loadedModel
                      ? "已載入"
                      : "沒有已載入模型"}
                </span>
                <span className="mt-1 block truncate text-sm font-medium">
                  {loadedModel ? modelLabel(loadedModel.id) : "選擇模型"}
                </span>
                {engine.status?.memory.active_gb != null && (
                  <span className="mt-0.5 block text-[11px] tabular-nums text-muted-foreground">
                    {number(engine.status.memory.active_gb)} /{" "}
                    {number(engine.status.memory.total_gb)} GB
                  </span>
                )}
              </Button>
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
            <div className="space-y-3 p-4">
              <Button
                variant={page === "settings" ? "secondary" : "ghost"}
                className="w-full justify-start"
                onClick={() => navigate("settings")}
              >
                <SettingsIcon size={16} />
                設定
              </Button>
              <p className="border-t border-border/60 px-2 pt-3 font-mono text-[10px] text-muted-foreground">
                {engine.status?.version
                  ? `yunshu ${engine.status.version}`
                  : "本機優先 · 真實引擎連線"}
              </p>
            </div>
          }
        />
        <main className="flex h-dvh min-w-0 flex-col lg:pl-64">
          <header className="flex min-h-14 shrink-0 items-center justify-between gap-2 border-b border-border/60 px-4 lg:px-6">
            <div className="flex items-center gap-2">
              <IconButton
                className="lg:hidden"
                icon={<Menu size={18} />}
                label="開啟導覽"
                onClick={() => setMenu(true)}
              />
              <span className="text-sm font-medium">{titles[page]}</span>
            </div>
            <div className="flex items-center gap-3">
              {engine.phase === "online" && engine.status && (
                <span className="hidden items-center gap-3 text-xs tabular-nums text-muted-foreground md:flex">
                  <span>
                    <span className="text-foreground">
                      {number(engine.status.requests.active, 0)}
                    </span>{" "}
                    req
                  </span>
                  <span>
                    <span className="text-foreground">
                      {number(
                        engine.status.throughput.live_decode_tps ??
                          engine.status.throughput.mean_decode_tps,
                      )}
                    </span>{" "}
                    tok/s
                  </span>
                </span>
              )}
              <Button
                size="sm"
                variant="ghost"
                className="hidden text-muted-foreground sm:inline-flex"
                onClick={() => setPalette(true)}
              >
                <Search size={13} />
                搜尋
                <Kbd>⌘K</Kbd>
              </Button>
              <StatusIndicator
                className="hidden text-xs text-muted-foreground sm:inline-flex"
                status={
                  engine.phase === "online"
                    ? "online"
                    : engine.phase === "connecting"
                      ? "away"
                      : "offline"
                }
              >
                {engine.phase === "online"
                  ? "已連線"
                  : engine.phase === "connecting"
                    ? "連線中"
                    : "未連線"}
              </StatusIndicator>
              <IconButton
                icon={dark ? <Sun size={16} /> : <Moon size={16} />}
                label={dark ? "切換淺色" : "切換深色"}
                onClick={() => setDark((v) => !v)}
              />
            </div>
          </header>
          {busy && (
            <div
              role="status"
              className="shrink-0 border-b border-border/60 px-5 py-2 text-xs text-muted-foreground"
            >
              {busy.startsWith("load:")
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
                          : "正在取消請求"}
              … 等待服務回應。
            </div>
          )}
          {notice && (
            <div
              role={notice.error ? "alert" : "status"}
              className="flex shrink-0 items-center justify-between gap-3 border-b border-border/60 px-5 py-3"
            >
              <p
                className={
                  "text-xs " +
                  (notice.error ? "text-error" : "text-muted-foreground")
                }
              >
                {notice.text}
              </p>
              <IconButton
                icon={<X size={13} />}
                label="關閉操作訊息"
                onClick={() => setNotice(null)}
              />
            </div>
          )}
          <div key={revision} className="flex min-h-0 flex-1 flex-col">
            {
              <div
                className={
                  engine.phase === "online"
                    ? "hidden"
                    : "shrink-0 px-4 pt-4 lg:px-6"
                }
              >
                <ConnectionState
                  engine={engine}
                  configure={() => navigate("settings")}
                />
              </div>
            }
            {page === "playground" ? (
              <Playground
                connection={connection}
                engine={engine}
                initialModel={testModel}
              />
            ) : (
              <div className="min-h-0 flex-1 overflow-y-auto p-4 lg:p-6">
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
            )}
          </div>
          <StatusPillBar
            ariaLabel="最近一筆請求"
            className="shrink-0 px-4 lg:px-6"
          >
            <StatusPill
              label={connectionText[engine.phase]}
              tone={connectionTone[engine.phase]}
              help={
                engine.phase === "online" && engine.status?.version
                  ? `引擎連線正常 · yunshu ${engine.status.version}`
                  : connectionHelp[engine.phase]
              }
            />
            {engine.phase === "online" && last && (
              <>
                {last.ttft_ms != null && (
                  <StatusPill
                    label="TTFT"
                    value={`${number(last.ttft_ms, 0)} ms`}
                    help={lastHelp(last, "首 token 延遲")}
                    dot={false}
                  />
                )}
                {last.decode_tps != null && (
                  <StatusPill
                    label="Decode"
                    value={`${number(last.decode_tps)} tok/s`}
                    help={lastHelp(last, "解碼速度")}
                    dot={false}
                  />
                )}
                {last.prefill_tps != null && (
                  <StatusPill
                    label="Prefill"
                    value={`${number(last.prefill_tps, 0)} tok/s`}
                    help={lastHelp(last, "預填速度")}
                    dot={false}
                  />
                )}
                {last.prompt_tokens > 0 && (
                  <StatusPill
                    label="快取"
                    value={`${number(last.cached_tokens, 0)} / ${number(last.prompt_tokens, 0)}`}
                    help={lastHelp(
                      last,
                      "命中前綴快取的 token 數 / 輸入 token 數",
                    )}
                    dot={false}
                  />
                )}
                {last.speculative?.mode && (
                  <StatusPill
                    label={String(last.speculative.mode).toUpperCase()}
                    value={
                      last.speculative.acceptance_rate != null
                        ? `接受 ${number(last.speculative.acceptance_rate * 100, 0)}%`
                        : undefined
                    }
                    tone="info"
                    help={lastHelp(
                      last,
                      `推測解碼${
                        last.speculative.rounds && last.completion_tokens
                          ? ` · 每輪 ${number(last.completion_tokens / last.speculative.rounds, 2)} tok`
                          : ""
                      }`,
                    )}
                  />
                )}
              </>
            )}
          </StatusPillBar>
        </main>
        <CommandPalette
          open={palette}
          onClose={() => {
            setPalette(false);
            setQuery("");
          }}
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

const connectionText = {
  connecting: "連線中",
  online: "運作中",
  offline: "離線",
  unauthorized: "未授權",
} as const;
const connectionTone = {
  connecting: "neutral",
  online: "success",
  offline: "danger",
  unauthorized: "warning",
} as const;
const connectionHelp = {
  connecting: "正在連線到本機引擎。",
  online: "引擎連線正常。",
  offline: "無法連線到引擎，請確認 Yunshu 服務正在執行。",
  unauthorized: "引擎拒絕了這組存取金鑰，請到設定更新。",
} as const;

/** The sentence behind a last-request pill: which request, when, how many tokens. */
function lastHelp(last: EngineLastRequest, what: string): string {
  return `${what} · 最近一筆請求 ${last.request_id} · ${clock(last.t * 1000)} · ${number(last.prompt_tokens, 0)} 輸入 / ${number(last.completion_tokens, 0)} 輸出`;
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
