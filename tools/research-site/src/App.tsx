import { useEffect, useState } from "react";
import { IconButton } from "@yuhuanowo/yunui";
import { Sidebar } from "@yuhuanowo/yunui/patterns";
import { YunUIProvider } from "@yuhuanowo/yunui/adapters";
import { BookOpen, Cpu, FileText, GitBranch, Home as HomeIcon, Layers, Menu, Moon, Scale, ScrollText, Sun, X } from "lucide-react";
import { useApi, useLiveStatus, type Meta } from "./api";
import { Home } from "./Home";
import { Lines } from "./Lines";
import { Decisions } from "./Decisions";
import { Ground } from "./Ground";
import { ParityPage } from "./Parity";
import { GpuqPage } from "./Gpuq";
import { Docs } from "./Docs";
import { NotFound, Status } from "./ui";

const titles: Record<string, string> = {
  "": "首頁",
  lines: "研究線",
  decisions: "使用者決策",
  ground: "事實基準",
  parity: "對手比較板",
  gpu: "GPU 佇列",
  docs: "文件",
};

export function parseRoute(hash: string): { page: string; docId: string | null; anchor: string } {
  const raw = hash.replace(/^#\/?/, "");
  const [path, query = ""] = raw.split("?");
  const [page, ...rest] = path.split("/");
  const anchor = new URLSearchParams(query).get("h") ?? "";
  let docId: string | null = null;
  if (page === "docs" && rest.length) {
    try {
      docId = rest.map(decodeURIComponent).join("/");
    } catch {
      docId = null;
    }
  }
  return { page, docId, anchor };
}

const adapters = { useT: () => (key: string) => key };

function stored(key: string, fallback: string) {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

export default function App() {
  const [route, setRoute] = useState(() => parseRoute(location.hash));
  const [menu, setMenu] = useState(false);
  const [dark, setDark] = useState(() => stored("yunshu.research.theme", matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") === "dark");
  const live = useLiveStatus();
  const meta = useApi<Meta>("/api/meta");
  useEffect(() => {
    const fn = () => {
      setRoute(parseRoute(location.hash));
      setMenu(false);
    };
    addEventListener("hashchange", fn);
    return () => removeEventListener("hashchange", fn);
  }, []);
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("yunshu.research.theme", dark ? "dark" : "light");
    } catch {}
  }, [dark]);
  const known = Object.hasOwn(titles, route.page);
  const nav = (href: string) => {
    location.hash = href === "/" ? "/" : href;
  };
  const m = meta.data;
  return (
    <YunUIProvider adapters={adapters}>
      <div className="h-dvh overflow-hidden bg-background" data-yunui-density="compact">
        <Sidebar
          appName="Yunshu"
          ariaLabel="內部文檔導覽"
          currentPath={route.page ? "/" + route.page : "/"}
          isOpen={menu}
          onClose={() => setMenu(false)}
          closeLabel="關閉導覽"
          onNavigate={nav}
          header={
            <div className="px-4 pb-2 pt-5">
              <div className="flex items-center gap-2.5 px-2">
                <Layers size={22} />
                <span className="flex-1 text-lg font-semibold">
                  Yunshu <span className="text-sm font-normal text-muted-foreground">內部文檔</span>
                </span>
                <IconButton className="lg:hidden" icon={<X size={17} />} label="關閉導覽" onClick={() => setMenu(false)} />
              </div>
            </div>
          }
          sections={[
            {
              title: "總覽",
              items: [
                { label: "首頁", href: "/", icon: HomeIcon },
                { label: "研究線", href: "/lines", icon: GitBranch },
                { label: "GPU 佇列", href: "/gpu", icon: Cpu },
                { label: "對手比較板", href: "/parity", icon: Scale },
              ],
            },
            {
              title: "正本",
              items: [
                { label: "使用者決策", href: "/decisions", icon: ScrollText },
                { label: "事實基準", href: "/ground", icon: BookOpen },
                { label: "文件", href: "/docs", icon: FileText },
              ],
            },
          ]}
          footer={<p className="px-6 py-4 text-[11px] text-muted-foreground">私有 · 唯讀 · 內容不進 git</p>}
        />
        <main className="flex h-dvh min-w-0 flex-col lg:pl-64">
          <header className="flex min-h-14 shrink-0 items-center justify-between gap-2 border-b border-border/60 px-4 lg:px-6">
            <div className="flex items-center gap-2">
              <IconButton className="lg:hidden" icon={<Menu size={18} />} label="開啟導覽" onClick={() => setMenu(true)} />
              <span className="text-sm font-medium">{titles[route.page] ?? "找不到頁面"}</span>
            </div>
            <div className="flex items-center gap-3">
              <span title={live ? "已連上即時更新（SSE）" : "即時更新中斷，頁面會每 30 秒重試"}>
                <Status tone={live ? "success" : "warning"}>{live ? "即時" : "離線"}</Status>
              </span>
              <IconButton icon={dark ? <Sun size={16} /> : <Moon size={16} />} label={dark ? "切換淺色" : "切換深色"} onClick={() => setDark((v) => !v)} />
            </div>
          </header>
          <div className="relative min-h-0 flex-1 overflow-y-auto p-4 pb-10 lg:p-6 lg:pb-10" id="rs-scroll">
            {m && !m.hasResearch && (
              <p className="mb-4 text-xs text-[var(--error)]">找不到 docs/research：{m.researchRoot}</p>
            )}
            <div key={route.page}>
              {route.page === "" && <Home meta={m} />}
              {route.page === "lines" && <Lines />}
              {route.page === "decisions" && <Decisions />}
              {route.page === "ground" && <Ground meta={m} />}
              {route.page === "parity" && <ParityPage />}
              {route.page === "gpu" && <GpuqPage />}
              {route.page === "docs" && <Docs id={route.docId} anchor={route.anchor} meta={m} />}
              {!known && <NotFound what={`沒有「${route.page}」這一頁。`} />}
            </div>
            <div className="rs-scroll-fade" />
          </div>
        </main>
      </div>
    </YunUIProvider>
  );
}
