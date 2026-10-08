import { useMemo, useState } from "react";
import { BarChart, Card, SegmentedSelect, Table, Tbody, Td, Th, Thead, Tr } from "@yuhuanowo/yunui";
import { StatCard } from "@yuhuanowo/yunui/patterns";
import { Boxes, CheckCircle2, CircleHelp, Trophy } from "lucide-react";
import { useApi, type Gap, type Parity as ParityT } from "./api";
import { Empty, Gate, Page, Section, Status, ago, num, stamp } from "./ui";

const METRIC_LABEL: Record<string, string> = {
  decode_cold_tps: "Decode（冷）tok/s",
  decode_warm_tps: "Decode（暖）tok/s",
  decode_turn2_tps: "Decode（第二輪）tok/s",
  prefill_cold_tps: "Prefill tok/s",
  ttft_cold_s: "TTFT（冷）秒",
  ttft_warm_s: "TTFT（暖）秒",
  followup_ttft_s: "Follow-up TTFT 秒",
  agentic_session_s: "Agentic session 秒",
  memory_idle_gib: "閒置記憶體",
  memory_peak_gib: "尖峰記憶體",
  energy_j_token: "每 token 能耗",
  accuracy_needle: "Needle 準確率",
};
const H2H_LABEL: Record<string, string> = { decode: "Decode 速度", ttft: "TTFT", followup_ttft: "Follow-up TTFT" };
const ENGINE_LABEL: Record<string, string> = { llamacpp: "llama.cpp", mlxlm: "mlx-lm", omlx: "oMLX", splash: "Splash", "tf-new": "TensorFold", mtplx: "MTPLX", strata: "Strata" };

/** 100 = as good as the best engine; lower = behind. Direction-aware. */
const perf = (g: Gap) => (g.higherIsBetter ? g.ours / g.best : g.best / g.ours) * 100;
const ctxLabel = (c: number) => (c >= 1024 ? `${c / 1024}K` : String(c));

function Charts({ gaps }: { gaps: Gap[] }) {
  const metrics = useMemo(() => [...new Set(gaps.map((g) => g.metric))], [gaps]);
  const [metric, setMetric] = useState("");
  const m = metrics.includes(metric) ? metric : metrics[0];
  const rows = gaps.filter((g) => g.metric === m);
  const data = rows.map((g) => ({
    id: `${g.ctx}-${g.kind}`,
    label: `${ctxLabel(g.ctx)} · ${g.kind}`,
    value: Math.round(perf(g) * 10) / 10,
    tone: perf(g) >= 100 ? ("success" as const) : perf(g) >= 90 ? ("warning" as const) : ("error" as const),
  }));
  return (
    <Section
      title="暫定差距：相對最佳引擎"
      hint="100% = 已是最佳；低於 100% = 落後。每格僅 1–3 次量測，是暫定值，不是結論。"
    >
      <div className="mb-4 flex flex-wrap gap-2">
        <SegmentedSelect value={m} onChange={setMetric} options={metrics.slice(0, 5).map((k) => ({ value: k, label: METRIC_LABEL[k] ?? k }))} />
      </div>
      {data.length === 0 ? (
        <Empty title="這個指標還沒有暫定資料" />
      ) : (
        <BarChart ariaLabel={`${METRIC_LABEL[m] ?? m} 相對最佳引擎的百分比`} data={data} formatValue={(v) => `${num(v, 1)}%`} />
      )}
      <div className="mt-4 overflow-x-auto">
        <Table scrollLabel="暫定差距明細" className="min-w-[560px]">
          <Thead>
            <Tr>
              <Th>情境</Th>
              <Th>Yunshu</Th>
              <Th>最佳</Th>
              <Th>最佳引擎</Th>
              <Th>相對最佳</Th>
              <Th>量測次數</Th>
            </Tr>
          </Thead>
          <Tbody>
            {rows.map((g) => (
              <Tr key={`${g.ctx}-${g.kind}`}>
                <Td className="text-xs">{ctxLabel(g.ctx)} · {g.kind}</Td>
                <Td className="text-xs">{num(g.ours, 3)}</Td>
                <Td className="text-xs">{num(g.best, 3)}</Td>
                <Td className="text-xs">{ENGINE_LABEL[g.best_engine] ?? g.best_engine}</Td>
                <Td className="text-xs">{num(perf(g), 1)}%</Td>
                <Td className="text-xs text-muted-foreground">Yunshu {g.reps["yunshu-new"] ?? 0} · 最佳 {g.reps[g.best_engine] ?? 0}</Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
      </div>
    </Section>
  );
}

function H2H({ d }: { d: ParityT }) {
  const engines = Object.keys(d.headToHead);
  const cats = [...new Set(engines.flatMap((e) => Object.keys(d.headToHead[e])))];
  return (
    <Section title="逐引擎對照" hint="勝 / 平 / 負（比較格數）。unknown（量測不足）不計入負。">
      <Table scrollLabel="逐引擎對照" className="min-w-[520px]">
        <Thead>
          <Tr>
            <Th>對手</Th>
            {cats.map((c) => (
              <Th key={c}>{H2H_LABEL[c] ?? c}</Th>
            ))}
          </Tr>
        </Thead>
        <Tbody>
          {engines.map((e) => (
            <Tr key={e}>
              <Td className="text-xs font-medium">{ENGINE_LABEL[e] ?? e}</Td>
              {cats.map((c) => {
                const x = d.headToHead[e][c];
                if (!x) return <Td key={c} className="text-xs text-muted-foreground" title="沒有可比較的格子">—</Td>;
                const tone = x.loss > x.win ? "error" : x.win > x.loss ? "success" : "muted";
                return (
                  <Td key={c} className="text-xs">
                    <Status tone={tone}>
                      <span className="text-foreground">
                        {x.win} / {x.tie} / {x.loss}
                      </span>
                      <span>共 {x.n} 格{x.provisional ? " · 暫定" : ""}</span>
                    </Status>
                  </Td>
                );
              })}
            </Tr>
          ))}
        </Tbody>
      </Table>
    </Section>
  );
}

export function ParityPage() {
  const api = useApi<ParityT>("/api/parity", ["research"]);
  return (
    <Page title="對手比較板" description="Yunshu 與其他引擎逐項比較；失敗關閉：資料不足一律是 unknown，不是通過。">
      <Gate api={api}>
        {(d) => (
          <>
            <Card className="p-5">
              <p className="text-sm font-semibold">{d.parity === d.total ? "全部持平或領先" : `持平 ${d.parity} / ${d.total} 項`}</p>
              <p className="mt-1 text-xs leading-5 text-muted-foreground">{d.summary}</p>
              <p className="mt-3 break-words font-mono text-[11px] leading-5 text-muted-foreground">{d.verdict}</p>
              <p className="mt-3 text-xs text-muted-foreground">
                board.json 更新於 {stamp(d.mtime)}（{ago(d.mtime)}）· <a className="underline underline-offset-4" href="#/docs/r/parityboard/BOARD.md">BOARD.md</a> · <a className="underline underline-offset-4" href="#/docs/r/parityboard/HANDOFF.md">HANDOFF</a>
              </p>
            </Card>
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              <StatCard compact valueFirst icon={CheckCircle2} label="持平的項目" value={`${d.parity}/${d.total}`} subtext={d.gateOpen ? "gate 已開" : "gate 未開"} />
              <StatCard compact valueFirst icon={Boxes} label="已有暫定資料" value={d.withData} subtext="不含記憶體與能耗" />
              <StatCard compact valueFirst icon={Trophy} label="Yunshu 目前最佳" value={d.bestOurs} subtext={`共 ${d.withData} 項暫定資料中`} />
              <StatCard compact valueFirst icon={CircleHelp} label="缺資料" value={d.missing ?? "—"} subtext="需要 ≥ 3 次獨立量測" />
            </div>
            <H2H d={d} />
            {d.gaps.length ? <Charts gaps={d.gaps} /> : <Empty title="還沒有暫定差距資料" description="需要至少一次完整量測。" />}
            <Section title="被排除的量測" hint={`${d.rejected.length} 格；另有 ${d.methodFlagged} 個記憶體格因量測方法不可比而標記`}>
              {d.rejected.length === 0 ? (
                <p className="text-xs text-muted-foreground">沒有被排除的格子。</p>
              ) : (
                <ul className="divide-y divide-border">
                  {d.rejected.map((r) => (
                    <li key={r.cell} className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 py-2 text-xs">
                      <span className="font-mono">{r.cell}</span>
                      <span className="text-muted-foreground">{r.reason}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Section>
            <p className="text-xs leading-5 text-muted-foreground">{d.policy}</p>
          </>
        )}
      </Gate>
    </Page>
  );
}
