import { useEffect, useMemo, useState } from "react";
import { Card, SearchInput } from "@yuhuanowo/yunui";
import { ArrowLeft } from "lucide-react";
import { useApi, type Doc, type DocMeta, type Hit, type Meta } from "./api";
import { Markdown } from "./Markdown";
import { Empty, ErrorState, Gate, Loading, Page, ago, stamp } from "./ui";

const href = (id: string) => `#/docs/${id.split("/").map(encodeURIComponent).join("/")}`;

function useDebounced<T>(v: T, ms: number): T {
  const [d, setD] = useState(v);
  useEffect(() => {
    const t = setTimeout(() => setD(v), ms);
    return () => clearTimeout(t);
  }, [v, ms]);
  return d;
}

function Viewer({ id, anchor, meta }: { id: string; anchor: string; meta: Meta | null }) {
  const api = useApi<Doc>(`/api/doc?id=${encodeURIComponent(id)}`, ["research", "codex"]);
  useEffect(() => {
    if (!api.data) return;
    if (!anchor) return void window.scrollTo?.(0, 0);
    const el = document.getElementById(anchor);
    if (el) el.scrollIntoView();
  }, [api.data?.id, anchor]); // eslint-disable-line react-hooks/exhaustive-deps
  if (api.error && !api.data) return <ErrorState error={api.error} retry={api.reload} title="打不開這份文件" />;
  if (!api.data) return <Loading />;
  const d = api.data;
  const plain = !/\.md$/i.test(id);
  return (
    <Card className="min-w-0 p-5 sm:p-6">
      <div className="mb-4 space-y-1 border-b border-border pb-3">
        <p className="break-all font-mono text-[11px] text-muted-foreground">{id.replace(/^r\//, "docs/research/").replace(/^c\//, "codex/")}</p>
        <p className="text-xs text-muted-foreground">
          修改於 {stamp(d.mtime)}（{ago(d.mtime)}）{d.truncated ? " · 檔案過大，只顯示前 2 MB" : ""}
        </p>
      </div>
      <Markdown docId={id} meta={meta} text={plain ? "```\n" + d.text.replace(/```/g, "'''") + "\n```" : d.text} />
    </Card>
  );
}

export function Docs({ id, anchor, meta }: { id: string | null; anchor: string; meta: Meta | null }) {
  const list = useApi<DocMeta[]>("/api/docs", ["research"]);
  const [q, setQ] = useState("");
  const dq = useDebounced(q.trim(), 200);
  const search = useApi<Hit[]>(dq ? `/api/search?q=${encodeURIComponent(dq)}` : null, ["research"]);
  const groups = useMemo(() => {
    const g = new Map<string, DocMeta[]>();
    for (const d of list.data ?? []) {
      const parts = d.id.slice(2).split("/");
      const key = parts.length > 1 ? parts[0] : "（根目錄）";
      g.set(key, [...(g.get(key) ?? []), d]);
    }
    return [...g].sort((a, b) => (a[0] === "（根目錄）" ? -1 : b[0] === "（根目錄）" ? 1 : a[0].localeCompare(b[0])));
  }, [list.data]);
  const showList = !id;
  return (
    <Page title="文件" description="docs/research 底下的所有 Markdown；連結會就地解析，搜尋涵蓋全文。">
      <div className="grid gap-5 lg:grid-cols-[320px_minmax(0,1fr)]">
        <div className={`${showList ? "" : "hidden lg:block"} min-w-0 space-y-3 lg:sticky lg:top-4 lg:self-start`}>
          <SearchInput aria-label="全文搜尋" placeholder="搜尋所有文件" value={q} onChange={setQ} />
          <Card className="max-h-[70vh] overflow-y-auto p-2">
            {dq ? (
              search.loading || (!search.data && !search.error) ? (
                <p className="p-3 text-xs text-muted-foreground">搜尋中…</p>
              ) : search.error ? (
                <p className="p-3 text-xs text-[var(--error)]">搜尋失敗：{search.error}</p>
              ) : search.data!.length === 0 ? (
                <p className="p-3 text-xs text-muted-foreground">沒有符合「{dq}」的文件。</p>
              ) : (
                <ul>
                  {search.data!.map((h) => (
                    <li key={h.id}>
                      <a href={href(h.id)} className="block rounded-lg px-3 py-2 transition-colors duration-100 hover:bg-muted">
                        <span className="block truncate text-xs font-medium">{h.title}</span>
                        <span className="block truncate font-mono text-[11px] text-muted-foreground">{h.id.slice(2)}</span>
                        <span className="mt-1 line-clamp-2 text-[11px] leading-4 text-muted-foreground">{h.snippet}</span>
                      </a>
                    </li>
                  ))}
                </ul>
              )
            ) : list.error && !list.data ? (
              <p className="p-3 text-xs text-[var(--error)]">無法載入文件清單：{list.error}</p>
            ) : !list.data ? (
              <p className="p-3 text-xs text-muted-foreground">載入中…</p>
            ) : (
              groups.map(([name, docs]) => (
                <details key={name} className="group" open={name === "（根目錄）" || docs.some((d) => d.id === id)}>
                  <summary className="flex cursor-pointer list-none items-center justify-between rounded-lg px-3 py-1.5 text-xs font-medium transition-colors duration-100 hover:bg-muted">
                    <span className="truncate">{name}</span>
                    <span className="text-muted-foreground">{docs.length}</span>
                  </summary>
                  <ul className="pb-1">
                    {docs.map((d) => (
                      <li key={d.id}>
                        <a
                          href={href(d.id)}
                          aria-current={d.id === id ? "page" : undefined}
                          className={`block truncate rounded-lg py-1.5 pl-6 pr-3 text-xs transition-colors duration-100 hover:bg-muted ${d.id === id ? "bg-muted font-medium" : "text-muted-foreground"}`}
                          title={d.id.slice(2)}
                        >
                          {d.title}
                        </a>
                      </li>
                    ))}
                  </ul>
                </details>
              ))
            )}
          </Card>
        </div>
        <div className="min-w-0">
          {id ? (
            <>
              <a href="#/docs" className="mb-3 inline-flex items-center gap-1 text-xs text-muted-foreground underline-offset-4 hover:underline lg:hidden">
                <ArrowLeft size={13} />
                文件清單
              </a>
              <Viewer id={id} anchor={anchor} meta={meta} />
            </>
          ) : (
            <div className="hidden lg:block">
              <Gate api={list}>
                {(d) => (
                  <Empty title="選一份文件" description={`左側共有 ${d.length} 份 Markdown；也可以直接搜尋全文。`} />
                )}
              </Gate>
            </div>
          )}
        </div>
      </div>
    </Page>
  );
}
