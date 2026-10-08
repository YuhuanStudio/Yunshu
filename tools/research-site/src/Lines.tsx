import { useMemo, useState } from "react";
import { Card, SearchInput, SegmentedSelect } from "@yuhuanowo/yunui";
import { FileText, ScrollText } from "lucide-react";
import { useApi, type Line, type Lines as LinesT } from "./api";
import { Empty, Gate, Page, Status, ago } from "./ui";

const rank = (l: Line) => (l.status === "ready" ? 0 : l.status === "ready-old" ? 1 : l.status === "open" ? 2 : 3);
const docHref = (id: string) => `#/docs/${id.split("/").map(encodeURIComponent).join("/")}`;

function LineCard({ l, now }: { l: Line; now: number }) {
  const tone = l.status === "ready" ? "success" : l.status === "ready-old" ? "warning" : l.status === "open" ? "muted" : "muted";
  const label = { ready: `待合併 ${l.ready_sha}`, "ready-old": `舊的待合併 ${l.ready_sha}`, open: "進行中", merged: "已合併" }[l.status];
  const worker = l.worker_live ? "codex 執行中" : l.registered.length ? `已登記：${l.registered.join("、")}` : null;
  return (
    <Card className="rs-in flex min-w-0 flex-col gap-3 p-4">
      <div className="flex items-start justify-between gap-3">
        <h3 className="min-w-0 truncate font-mono text-sm font-semibold" title={l.branch}>
          {l.branch}
        </h3>
        <Status tone={tone}>{label}</Status>
      </div>
      <p className="line-clamp-2 min-h-[2.5rem] text-xs leading-5 text-muted-foreground" title={l.commit_subject}>
        {l.commit_subject || "沒有 commit"}
      </p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
        <dt className="text-muted-foreground">相對 main</dt>
        <dd className="font-mono">
          +{l.ahead} / −{l.behind}
        </dd>
        <dt className="text-muted-foreground">最後 commit</dt>
        <dd>{l.commit_ts ? ago(l.commit_ts * 1000, now) : "—"}</dd>
        <dt className="text-muted-foreground">Worker</dt>
        <dd>{worker ? <Status tone="success">{worker}</Status> : <span className="text-muted-foreground">無</span>}</dd>
        <dt className="text-muted-foreground">gpuq</dt>
        <dd>{l.gpuq_running || l.gpuq_pending ? `${l.gpuq_running} 執行 · ${l.gpuq_pending} 排隊` : <span className="text-muted-foreground">無</span>}</dd>
      </dl>
      {l.report_head && (
        <p className="line-clamp-2 text-xs leading-5" title={l.report_head}>
          {l.report_head}
        </p>
      )}
      <div className="mt-auto flex flex-wrap gap-x-4 gap-y-1 pt-1 text-xs">
        {l.handoff ? (
          <a className="inline-flex items-center gap-1 underline underline-offset-4" href={docHref(l.handoff)}>
            <FileText size={12} />
            HANDOFF
          </a>
        ) : (
          <span className="text-muted-foreground" title="這條線沒有 HANDOFF.md">無 HANDOFF</span>
        )}
        {l.report_id ? (
          <a className="inline-flex items-center gap-1 underline underline-offset-4" href={docHref(l.report_id)}>
            <ScrollText size={12} />
            最新報告
          </a>
        ) : (
          <span className="text-muted-foreground" title="找不到 worker 報告">無報告</span>
        )}
      </div>
    </Card>
  );
}

export function Lines() {
  const api = useApi<LinesT>("/api/lines", ["research", "jobs", "codex"]);
  const [filter, setFilter] = useState("all");
  const [q, setQ] = useState("");
  const now = Date.now();
  const shown = useMemo(() => {
    const l = api.data?.lines ?? [];
    const needle = q.trim().toLowerCase();
    return l
      .filter((x) => (filter === "all" ? true : filter === "ready" ? x.status.startsWith("ready") : filter === "open" ? x.status === "open" : x.status === "merged"))
      .filter((x) => !needle || `${x.branch} ${x.commit_subject} ${x.report_head}`.toLowerCase().includes(needle))
      .sort((a, b) => rank(a) - rank(b) || (b.commit_ts ?? 0) - (a.commit_ts ?? 0));
  }, [api.data, filter, q]);
  return (
    <Page title="研究線" description="每個 worktree 分支一張卡；資料來自 research_index.py --json 與 gpuq 佇列。">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SearchInput className="w-full sm:max-w-xs" aria-label="搜尋研究線" placeholder="分支、commit 或報告" value={q} onChange={setQ} />
        <SegmentedSelect
          value={filter}
          onChange={setFilter}
          options={[
            { value: "all", label: "全部" },
            { value: "open", label: "進行中" },
            { value: "ready", label: "待合併" },
            { value: "merged", label: "已合併" },
          ]}
        />
      </div>
      <Gate api={api}>
        {(d) =>
          !d.ok ? (
            <Empty title="無法取得各線資料" description={d.error} />
          ) : shown.length === 0 ? (
            <Empty title="沒有符合的研究線" description={q || filter !== "all" ? "試著清除搜尋或改選「全部」。" : "目前沒有任何 worktree 分支。"} />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {shown.map((l) => (
                <LineCard key={l.branch} l={l} now={now} />
              ))}
            </div>
          )
        }
      </Gate>
    </Page>
  );
}
