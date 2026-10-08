import type { ReactNode } from "react";
import { Alert, Button, Card, EmptyState, Skeleton } from "@yuhuanowo/yunui";
import { AlertTriangle, FileQuestion, Inbox } from "lucide-react";
import type { Api } from "./api";

export const num = (v: number | null | undefined, d = 1) =>
  v == null || !Number.isFinite(v) ? "—" : v.toLocaleString("zh-TW", { maximumFractionDigits: d });

export function dur(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "—";
  s = Math.max(0, s);
  if (s < 90) return `${Math.round(s)} 秒`;
  if (s < 5400) return `${Math.round(s / 60)} 分`;
  if (s < 172800) return `${(s / 3600).toFixed(1)} 小時`;
  return `${Math.round(s / 86400)} 天`;
}
export const ago = (ms: number | null | undefined, now = Date.now()) => (ms == null ? "—" : `${dur((now - ms) / 1000)}前`);
export const clock = (ms: number) =>
  new Date(ms).toLocaleTimeString("zh-TW", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
export const stamp = (ms: number) =>
  new Date(ms).toLocaleString("zh-TW", { hour12: false, month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });

export function Dot({ tone }: { tone: "success" | "warning" | "error" | "muted" }) {
  const c = { success: "bg-[var(--success)]", warning: "bg-[var(--warning)]", error: "bg-[var(--error)]", muted: "bg-[var(--text-tertiary)]" }[tone];
  return <span aria-hidden className={`inline-block size-1.5 shrink-0 rounded-full ${c}`} />;
}

export function Status({ tone, children }: { tone: "success" | "warning" | "error" | "muted"; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
      <Dot tone={tone} />
      {children}
    </span>
  );
}

export function Page({ title, description, actions, children }: { title: string; description?: ReactNode; actions?: ReactNode; children: ReactNode }) {
  return (
    <div className="rs-route mx-auto w-full max-w-7xl space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
          {description && <p className="mt-1 text-xs text-muted-foreground">{description}</p>}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {children}
    </div>
  );
}

export function Section({ title, hint, children, className = "" }: { title: string; hint?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <Card className={`min-w-0 p-5 ${className}`}>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2 className="text-sm font-semibold">{title}</h2>
        {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
      </div>
      {children}
    </Card>
  );
}

export function Loading() {
  return (
    <div className="space-y-3" aria-busy="true" aria-label="載入中">
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-48 w-full" />
    </div>
  );
}

export function ErrorState({ error, retry, title = "無法載入" }: { error: string; retry?: () => void; title?: string }) {
  return (
    <Card className="p-5">
      <EmptyState
        icon={<AlertTriangle size={20} />}
        title={title}
        description={error.startsWith("not found") ? "找不到這個檔案；它可能被移動、封存或尚未建立。" : `伺服器回應：${error}`}
        action={
          retry ? (
            <Button size="sm" variant="secondary" onClick={retry}>
              重試
            </Button>
          ) : undefined
        }
      />
    </Card>
  );
}

export function Empty({ title, description }: { title: string; description?: string }) {
  return (
    <Card className="p-5">
      <EmptyState icon={<Inbox size={20} />} title={title} description={description} />
    </Card>
  );
}

export function NotFound({ what }: { what: string }) {
  return (
    <Card className="p-5">
      <EmptyState icon={<FileQuestion size={20} />} title="找不到頁面" description={what} />
    </Card>
  );
}

/** Render a loaded API result with the standard loading / error / stale-data handling. */
export function Gate<T>({ api, children, retry = true }: { api: Api<T>; children: (data: T) => ReactNode; retry?: boolean }) {
  if (api.data) {
    return (
      <>
        {api.error && (
          <Alert variant="warning" className="mb-4">
            資料暫時無法更新（{api.error}），以下是上次成功的內容。
          </Alert>
        )}
        {children(api.data)}
      </>
    );
  }
  if (api.error) return <ErrorState error={api.error} retry={retry ? api.reload : undefined} />;
  return <Loading />;
}
