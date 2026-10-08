import { useMemo, useState } from "react";
import { Card, CustomSelect, SearchInput, SegmentedSelect, Switch, Table, Tbody, Td, Th, Thead, Tr } from "@yuhuanowo/yunui";
import { useApi, type Decision } from "./api";
import { Empty, Gate, Page } from "./ui";

type Resp = { mtime: number; rows: Decision[] };

export function Decisions() {
  const api = useApi<Resp>("/api/decisions", ["research"]);
  const [q, setQ] = useState("");
  const [section, setSection] = useState("all");
  const [range, setRange] = useState("all");
  const [old, setOld] = useState(true);
  const rows = api.data?.rows ?? [];
  const sections = useMemo(() => {
    const m = new Map<string, string>();
    rows.forEach((r) => m.set(r.section, r.sectionTitle));
    return [...m].map(([value, label]) => ({ value, label: `${value}. ${label}` }));
  }, [rows]);
  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const cut = range === "all" ? "" : new Date(Date.now() - Number(range) * 86400000).toISOString().slice(0, 10);
    return rows
      .filter((r) => section === "all" || r.section === section)
      .filter((r) => old || !r.superseded)
      .filter((r) => !cut || (/^\d{4}-\d{2}-\d{2}/.test(r.date) ? r.date.slice(0, 10) >= cut : false))
      .filter((r) => !needle || `${r.date} ${r.decision} ${r.quote} ${r.source}`.toLowerCase().includes(needle))
      .sort((a, b) => b.date.localeCompare(a.date));
  }, [rows, q, section, range, old]);
  return (
    <Page title="使用者決策" description="只記使用者本人的決定，附日期與原話；後來的決策明寫取代前者。">
      <div className="flex flex-wrap items-center gap-3">
        <SearchInput className="w-full sm:max-w-xs" aria-label="搜尋決策" placeholder="決策、原話或來源" value={q} onChange={setQ} />
        <CustomSelect
          className="w-full sm:w-56"
          value={section}
          onChange={setSection}
          options={[{ value: "all", label: "全部章節" }, ...sections]}
        />
        <SegmentedSelect
          value={range}
          onChange={setRange}
          options={[
            { value: "all", label: "全部日期" },
            { value: "30", label: "近 30 天" },
            { value: "7", label: "近 7 天" },
            { value: "2", label: "近 2 天" },
          ]}
        />
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <Switch checked={old} onCheckedChange={setOld} label="顯示已取代的決策" />
          顯示已取代
        </label>
        <span className="ml-auto text-xs text-muted-foreground">
          {shown.length} / {rows.length} 筆
        </span>
      </div>
      <Gate api={api}>
        {() =>
          shown.length === 0 ? (
            <Empty title="沒有符合的決策" description="試著清除搜尋、日期或章節篩選。" />
          ) : (
            <Card className="overflow-hidden">
              <Table scrollLabel="使用者決策" className="min-w-[900px]">
                <Thead>
                  <Tr>
                    <Th className="w-28">日期</Th>
                    <Th className="w-16">章節</Th>
                    <Th>決策</Th>
                    <Th className="w-[28%]">原話／依據</Th>
                    <Th className="w-40">來源</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {shown.map((r, i) => (
                    <Tr key={`${r.date}-${i}`} className={r.superseded ? "opacity-60" : ""}>
                      <Td className="whitespace-nowrap align-top text-xs">{r.date}</Td>
                      <Td className="whitespace-nowrap align-top text-xs text-muted-foreground" title={r.sectionTitle}>
                        {r.section}
                      </Td>
                      <Td className="align-top text-xs leading-5">
                        {r.superseded && <span className="mr-1 text-muted-foreground">[已取代]</span>}
                        {r.decision}
                      </Td>
                      <Td className="align-top text-xs leading-5 text-muted-foreground">{r.quote}</Td>
                      <Td className="align-top text-xs text-muted-foreground">{r.source}</Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </Card>
          )
        }
      </Gate>
    </Page>
  );
}
