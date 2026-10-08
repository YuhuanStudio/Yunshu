import { useState } from "react";
import { BarChart, Card, SegmentedSelect, Table, Tbody, Td, Th, Thead, Tr } from "@yuhuanowo/yunui";
import { StatCard } from "@yuhuanowo/yunui/patterns";
import { AlarmClock, Cpu, Hourglass, Trash2 } from "lucide-react";
import { useApi, type Gpuq as GpuqT, type JobRow } from "./api";
import { Empty, Gate, Page, Section, Status, dur, num } from "./ui";

const flags = (j: JobRow) => [j.gate && "gate", j.short && "short"].filter(Boolean).join(" · ") || "—";

function JobTable({ rows, kind }: { rows: JobRow[]; kind: "running" | "pending" }) {
  return (
    <Table scrollLabel={kind === "running" ? "執行中的 gpuq 工作" : "排隊中的 gpuq 工作"} className="min-w-[640px]">
      <Thead>
        <Tr>
          <Th className="w-16">優先</Th>
          <Th>工作</Th>
          <Th className="w-24">研究線</Th>
          <Th className="w-28">{kind === "running" ? "已執行" : "已等待"}</Th>
          <Th className="w-28">狀態</Th>
          <Th className="w-24">標記</Th>
        </Tr>
      </Thead>
      <Tbody>
        {rows.map((j) => (
          <Tr key={j.id}>
            <Td className="text-xs">p{j.priority}</Td>
            <Td className="max-w-[420px] truncate font-mono text-xs" title={j.label}>{j.label}</Td>
            <Td className="text-xs">{j.line}</Td>
            <Td className="whitespace-nowrap text-xs">{dur(kind === "running" ? j.runningS : j.waitS)}</Td>
            <Td className="text-xs">
              <Status tone={kind === "running" ? "success" : "muted"}>{j.display}</Status>
            </Td>
            <Td className="text-xs text-muted-foreground">{flags(j)}</Td>
          </Tr>
        ))}
      </Tbody>
    </Table>
  );
}

export function GpuqPage() {
  const api = useApi<GpuqT>("/api/gpuq", ["jobs"]);
  const [win, setWin] = useState("1");
  return (
    <Page title="GPU 佇列" description="唯讀；資料來自 gpuq 的 jobs 目錄（只取白名單欄位，不含指令與環境變數）。">
      <Gate api={api}>
        {(d) => {
          const s = win === "1" ? d.stats1h : d.stats24h;
          const lines = Object.entries(d.byLine)
            .filter(([, v]) => v.pending)
            .sort((a, b) => b[1].pending - a[1].pending)
            .slice(0, 12);
          const pct = s.gpuMin ? (100 * s.wastedMin) / s.gpuMin : 0;
          return (
            <>
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                <StatCard compact valueFirst icon={Cpu} label="執行中" value={d.running.length} subtext={d.running[0] ? `${d.running[0].line} · ${dur(d.running[0].runningS)}` : "GPU 閒置"} />
                <StatCard compact valueFirst icon={Hourglass} label="排隊中" value={d.pending.length} subtext={Object.entries(d.byPriority).map(([p, n]) => `${p}:${n}`).join(" · ") || "佇列是空的"} />
                <StatCard compact valueFirst icon={AlarmClock} label={`近 ${s.windowH} 小時 GPU`} value={<>{num(s.gpuMin, 0)}<span className="ml-1 text-xs font-normal text-muted-foreground">分鐘</span></>} subtext={`${s.jobs} 個工作結束`} />
                <StatCard compact valueFirst icon={Trash2} label={`近 ${s.windowH} 小時浪費`} value={<>{num(s.wastedMin, 0)}<span className="ml-1 text-xs font-normal text-muted-foreground">分鐘</span></>} subtext={s.gpuMin ? `佔 ${num(pct, 1)}%（失敗、逾時、被打擾的計時）` : "沒有資料"} />
              </div>
              <Section title="執行中" hint={`${d.running.length} 個`}>
                {d.running.length ? <JobTable rows={d.running} kind="running" /> : <Empty title="目前沒有執行中的工作" description="GPU 現在是閒置的。" />}
              </Section>
              <Section title="排隊中" hint={`${d.pending.length} 個；依優先序，同優先序先到先服務`}>
                {d.pending.length ? (
                  <div className="max-h-[480px] overflow-y-auto">
                    <JobTable rows={d.pending} kind="pending" />
                  </div>
                ) : (
                  <Empty title="佇列是空的" />
                )}
              </Section>
              <div className="grid gap-5 lg:grid-cols-2">
                <Section title="各線排隊數" hint="前 12 條">
                  {lines.length ? (
                    <BarChart ariaLabel="各研究線排隊中的工作數" data={lines.map(([k, v]) => ({ id: k, label: k, value: v.pending }))} formatValue={(v) => `${v}`} />
                  ) : (
                    <Empty title="沒有排隊的工作" />
                  )}
                </Section>
                <Section
                  title="各線用量與浪費"
                  hint={<SegmentedSelect value={win} onChange={setWin} options={[{ value: "1", label: "近 1 小時" }, { value: "24", label: "近 24 小時" }]} />}
                >
                  {s.perLine.length ? (
                    <div className="max-h-[360px] overflow-y-auto">
                      <Table scrollLabel="各線 GPU 用量" className="min-w-[360px]">
                        <Thead>
                          <Tr>
                            <Th>研究線</Th>
                            <Th>工作</Th>
                            <Th>失敗</Th>
                            <Th>GPU 分</Th>
                            <Th>浪費分</Th>
                          </Tr>
                        </Thead>
                        <Tbody>
                          {s.perLine.map((r) => (
                            <Tr key={r.line}>
                              <Td className="text-xs">{r.line}</Td>
                              <Td className="text-xs">{r.jobs}</Td>
                              <Td className="text-xs">{r.bad || "—"}</Td>
                              <Td className="text-xs">{num(r.gpuMin, 1)}</Td>
                              <Td className="text-xs">{r.wastedMin ? num(r.wastedMin, 1) : "—"}</Td>
                            </Tr>
                          ))}
                        </Tbody>
                      </Table>
                    </div>
                  ) : (
                    <Empty title="這段時間沒有結束的工作" />
                  )}
                </Section>
              </div>
            </>
          );
        }}
      </Gate>
    </Page>
  );
}
