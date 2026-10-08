import { Card } from "@yuhuanowo/yunui";
import { useApi, type Doc, type Meta } from "./api";
import { Markdown } from "./Markdown";
import { Gate, Page, ago, stamp } from "./ui";

export function Ground({ meta }: { meta: Meta | null }) {
  const api = useApi<Doc>("/api/doc?id=r/GROUND_TRUTH.md", ["research"]);
  return (
    <Page title="事實基準" description="由源碼直接推導的已驗證事實；與其他文檔矛盾時以此為準。">
      <Gate api={api}>
        {(d) => (
          <Card className="p-5 sm:p-6">
            <p className="mb-4 border-b border-border pb-3 text-xs text-muted-foreground">
              GROUND_TRUTH.md · 修改於 {stamp(d.mtime)}（{ago(d.mtime)}）
            </p>
            <Markdown docId="r/GROUND_TRUTH.md" meta={meta} text={d.text} />
          </Card>
        )}
      </Gate>
    </Page>
  );
}
