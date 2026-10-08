import { useMemo } from "react";
import { Alert, Button, Card } from "@yuhuanowo/yunui";
import { StatCard } from "@yuhuanowo/yunui/patterns";
import { Activity, FileText, GitBranch, Scale, RefreshCw } from "lucide-react";
import { useApi, type Gpuq, type IndexDoc, type Lines, type Meta, type Parity } from "./api";
import { Markdown } from "./Markdown";
import { Gate, Page, Status, ago, stamp } from "./ui";

const LINES_BLOCK = /<!-- auto:lines -->[\s\S]*?<!-- \/auto:lines -->/;

export function Home({ meta }: { meta: Meta | null }) {
  const index = useApi<IndexDoc>("/api/index", ["research"]);
  const lines = useApi<Lines>("/api/lines", ["research", "jobs", "codex"]);
  const gpu = useApi<Gpuq>("/api/gpuq", ["jobs"]);
  const parity = useApi<Parity>("/api/parity", ["research"]);
  const counts = useMemo(() => {
    const l = lines.data?.lines ?? [];
    return {
      open: l.filter((x) => x.status === "open").length,
      ready: l.filter((x) => x.status === "ready").length,
      merged: l.filter((x) => x.status === "merged").length,
      live: l.filter((x) => x.worker_live || x.registered.length).length,
    };
  }, [lines.data]);
  return (
    <Page
      title="雲樞內部文檔"
      description="docs/research 的即時唯讀檢視；檔案一改、佇列一動就自動更新。"
      actions={
        <Button size="sm" variant="ghost" onClick={() => (index.reload(), lines.reload(), gpu.reload(), parity.reload())}>
          <RefreshCw size={13} />
          重新整理
        </Button>
      }
    >
      <Gate api={index}>
        {(d) => (
          <>
            {d.stale && (
              <Alert variant="warning" title="index-stale">
                {d.ageMin == null
                  ? "INDEX.md 沒有自動區塊時間戳，無法判斷新舊。"
                  : `INDEX.md 的自動區塊已 ${Math.round(d.ageMin)} 分鐘沒有重生（超過 60 分鐘）。下方「各線」與「GPU 佇列」是即時資料，不受影響；INDEX 文字請執行 scripts/dev/research_index.py。`}
              </Alert>
            )}
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              <StatCard
                compact
                valueFirst
                icon={GitBranch}
                label="研究線"
                value={lines.data ? lines.data.lines.length : "—"}
                subtext={lines.data ? `進行中 ${counts.open} · 待合併 ${counts.ready} · 已合併 ${counts.merged}` : lines.error ? "無法取得" : "載入中"}
              />
              <StatCard
                compact
                valueFirst
                icon={Activity}
                label="GPU 佇列"
                value={gpu.data ? `${gpu.data.running.length} / ${gpu.data.pending.length}` : "—"}
                subtext="執行中 / 排隊中"
              />
              <StatCard
                compact
                valueFirst
                icon={Scale}
                label="對手比較板"
                value={parity.data ? `${parity.data.parity}/${parity.data.total}` : "—"}
                subtext={parity.data ? `已有暫定資料 ${parity.data.withData} 項；unknown 不等於落後` : parity.error ? "無法取得" : "載入中"}
              />
              <StatCard
                compact
                valueFirst
                icon={FileText}
                label="INDEX 更新"
                value={d.ageMin == null ? "—" : `${Math.round(d.ageMin)} 分鐘前`}
                subtext={
                  <Status tone={d.stale ? "warning" : "success"}>{d.stale ? "index-stale（> 60 分鐘）" : "新鮮"}</Status>
                }
              />
            </div>
            <Card className="p-5 sm:p-6">
              <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2 text-xs text-muted-foreground">
                <span>INDEX.md · 檔案修改 {stamp(d.mtime)}（{ago(d.mtime)}）</span>
                <a href="#/lines" className="underline underline-offset-4">各線即時狀態在「研究線」頁</a>
              </div>
              <Markdown
                docId="r/INDEX.md"
                meta={meta}
                text={d.text.replace(LINES_BLOCK, "> 各線表格已改為即時資料，請看「研究線」頁。")}
              />
            </Card>
          </>
        )}
      </Gate>
    </Page>
  );
}
