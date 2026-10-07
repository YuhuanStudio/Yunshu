import { useEffect, useState } from "react";
import { Button, IconButton } from "@yuhuanowo/yunui";
import { Sidebar } from "@yuhuanowo/yunui/patterns";
import { YunUIProvider } from "@yuhuanowo/yunui/adapters";
import {
  Activity,
  Box,
  Code2,
  Gauge,
  Layers,
  Menu,
  MessageSquare,
  Moon,
  Settings as SettingsIcon,
  Sun,
  X,
} from "lucide-react";
import { useEngine } from "./useEngine";
import type { Connection } from "./api";
import { ConnectionState } from "./ui";
import { Dashboard } from "./Dashboard";
import { Models } from "./Models";
import { Requests } from "./Requests";
import { Settings, ApiView } from "./Settings";
import { Playground } from "./Playground";
const titles: Record<string, string> = {
  overview: "引擎總覽",
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
      await action();
      await engine.refresh();
      setNotice({
        error: false,
        text: key.startsWith("cancel:")
          ? "已送出取消請求。"
          : "操作已完成。",
      });
    } catch (e) {
      setNotice({
        error: true,
        text: e instanceof Error ? e.message : String(e),
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
            <div className="space-y-5 px-4 pb-2 pt-5">
              <div className="flex items-center gap-2.5 px-2">
                <Layers size={22} />
                <span className="flex-1 text-lg font-semibold">
                  Yunshu{" "}
                  <span className="text-sm font-normal text-muted-foreground">
                    雲舒
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
                className="w-full justify-start"
                onClick={() => navigate("models")}
              >
                <Box size={16} />
                管理模型
              </Button>
            </div>
          }
          sections={[
            {
              title: "推理引擎",
              items: [
                { label: "引擎總覽", href: "/overview", icon: Gauge },
                { label: "模型庫", href: "/models", icon: Box },
                { label: "請求與效能", href: "/requests", icon: Activity },
                { label: "API 接入", href: "/api", icon: Code2 },
              ],
            },
            {
              title: "工具",
              items: [
                { label: "推理測試", href: "/playground", icon: MessageSquare },
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
              <p className="border-t border-border/60 px-2 pt-4 text-[10px] text-muted-foreground">
                本機優先 · 真實引擎連線
              </p>
            </div>
          }
        />
        <main className="flex h-dvh min-w-0 flex-col lg:pl-64">
          <header className="flex min-h-16 shrink-0 items-center justify-between gap-2 border-b border-border/60 px-4 sm:px-7">
            <div className="flex items-center gap-2">
              <IconButton
                className="lg:hidden"
                icon={<Menu size={18} />}
                label="開啟導覽"
                onClick={() => setMenu(true)}
              />
              <span className="text-sm font-medium">{titles[page]}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="hidden text-xs text-muted-foreground sm:inline">
                {engine.phase === "online"
                  ? "已連線"
                  : engine.phase === "connecting"
                    ? "連線中"
                    : "未連線"}
              </span>
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
                    : "shrink-0 px-4 pt-4 sm:px-7"
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
              <div className="min-h-0 flex-1 overflow-y-auto">
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
                  />
                )}
                {page === "api" && <ApiView connection={connection} />}
              </div>
            )}
          </div>
        </main>
      </div>
    </YunUIProvider>
  );
}
