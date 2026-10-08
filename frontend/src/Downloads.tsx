import { useEffect, useState } from "react";
import { useRouteAction } from "./useRouteAction";
import { t } from "./i18n/index.ts";
import {
  Button,
  Card,
  EmptyState,
  Input,
  Label,
  Progress,
  StatusIndicator,
} from "@yuhuanowo/yunui";
import { Banner, DashboardPage, PageHeader } from "@yuhuanowo/yunui/patterns";
import { Download, RotateCcw, X } from "lucide-react";
import { ApiError, type Connection } from "./api";
import {
  cancelDownload,
  diskShortfall,
  isActive,
  parsePatterns,
  REPO_RE,
  startDownload,
  type DownloadJob,
  type DownloadRequest,
} from "./admin-models-api";
import { useDownloads } from "./admin-hooks";
import { bytesText, rateText } from "./byte-format";
import { detailText } from "./errors";
import { ByteValue } from "./ByteValue";
import { Reasoned } from "./Reasoned";
import {
  elapsed,
  fixed,
  SectionCard,
  UnavailableNotice,
  type Engine,
} from "./ui";

const jobStatus = (j: DownloadJob) =>
  j.state === "failed"
    ? "busy"
    : j.state === "running" || j.state === "queued"
      ? "away"
      : j.state === "done"
        ? "online"
        : "neutral";

const jobLabel = (j: DownloadJob) =>
  j.state === "queued"
    ? t("downloads.state.queued")
    : j.state === "running"
      ? t("downloads.state.running")
      : j.state === "done"
        ? t("downloads.state.done")
        : j.state === "cancelled"
          ? t("downloads.state.cancelled")
          : t("downloads.state.failed");

export function percentDone(j: DownloadJob): number | null {
  return j.bytesTotal && j.bytesTotal > 0 && j.bytesDone != null
    ? Math.max(0, Math.min(100, (j.bytesDone / j.bytesTotal) * 100))
    : null;
}

function JobRow({
  job,
  busy,
  onCancel,
  onResume,
}: {
  job: DownloadJob;
  busy: boolean;
  onCancel: (j: DownloadJob) => void;
  onResume: (j: DownloadJob) => void;
}) {
  const pct = percentDone(job),
    active = isActive(job),
    resumable = job.state === "cancelled" || job.state === "failed";
  return (
    <div
      className="space-y-2 rounded-md px-3 py-3"
      data-testid="download-row"
      data-state={job.state}
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="min-w-0 flex-1 basis-56">
          <p className="break-all text-sm font-medium">{job.repo}</p>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
            <StatusIndicator status={jobStatus(job)}>
              <span className="text-foreground">{jobLabel(job)}</span>
            </StatusIndicator>
            {job.revision && <span>{job.revision}</span>}
            {job.patterns.length > 0 && (
              <span className="font-mono">{job.patterns.join(", ")}</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-1">
          {active && (
            <Button
              size="sm"
              variant="secondary"
              disabled={busy}
              onClick={() => onCancel(job)}
            >
              <X size={12} />
              {t("downloads.cancel")}
            </Button>
          )}
          {resumable && (
            <Reasoned reason={busy ? t("downloads.busyReason") : null}>
              <Button
                size="sm"
                variant="secondary"
                disabled={busy}
                onClick={() => onResume(job)}
              >
                <RotateCcw size={12} />
                {t("downloads.resume")}
              </Button>
            </Reasoned>
          )}
          {job.state === "done" && job.registered && (
            <a
              className="text-sm underline underline-offset-2"
              href={`#/models/${encodeURIComponent(job.repo)}`}
            >
              {t("downloads.openModel")}
            </a>
          )}
        </div>
      </div>
      {(active || job.state === "done") && (
        <div className="space-y-1.5">
          {pct != null ? (
            <Progress
              className="h-1.5"
              value={pct}
              label={t("downloads.progressAria", { repo: job.repo })}
            />
          ) : (
            active && (
              <Progress
                className="h-1.5"
                indeterminate
                label={t("downloads.progressAria", { repo: job.repo })}
              />
            )
          )}
          <p
            className="flex flex-wrap items-baseline gap-x-4 gap-y-0.5 text-xs text-muted-foreground tabular-nums"
            data-testid="download-stats"
          >
            <span>
              <ByteValue bytes={job.bytesDone} className="text-foreground" />
              {" / "}
              <ByteValue bytes={job.bytesTotal} />
              {pct != null && ` · ${fixed(pct, 0)}%`}
            </span>
            {active && (
              <>
                <span title={t("downloads.rateTitle")}>
                  {rateText(job.rateBps)}
                </span>
                <span>
                  {t("downloads.eta", {
                    time: job.etaS == null ? "—" : elapsed(job.etaS),
                  })}
                </span>
              </>
            )}
            {job.filesTotal != null && job.filesTotal > 0 && (
              <span>
                {t("downloads.files", {
                  done: job.filesDone ?? 0,
                  total: job.filesTotal,
                })}
              </span>
            )}
          </p>
          {active && job.activeFiles.length > 0 && (
            <p className="truncate font-mono text-xs text-muted-foreground">
              {job.activeFiles.join(", ")}
            </p>
          )}
        </div>
      )}
      {job.alreadyPresent && (
        <p className="text-xs text-muted-foreground">
          {t("downloads.alreadyPresent")}
        </p>
      )}
      {job.state === "done" && !job.registered && !job.alreadyPresent && (
        <p className="text-xs text-muted-foreground">
          {t("downloads.notRegistered")}
        </p>
      )}
      {job.error && (
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer text-error">
            {t("downloads.failedSummary")}
          </summary>
          <p className="mt-1 break-words">{job.error}</p>
        </details>
      )}
    </div>
  );
}

export default function Downloads({
  connection,
  engine,
}: {
  connection: Connection;
  engine: Engine;
}) {
  const online = engine.status != null;
  const polled = useDownloads(connection, online);
  const [repo, setRepo] = useState(""),
    [revision, setRevision] = useState(""),
    [patterns, setPatterns] = useState(""),
    [pending, setPending] = useState(false),
    [notice, setNotice] = useState(""),
    [error, setError] = useState<ApiError | Error | null>(null);
  const data = polled.data;
  // Palette verb `#/downloads?action=new[&model=repo]`: prefill the repo and focus the field.
  const [focusRepo, setFocusRepo] = useState(0);
  useRouteAction("downloads", (action, q) => {
    if (action !== "new") return;
    const m = q.get("model");
    if (m) setRepo(m);
    setFocusRepo((n) => n + 1);
  });
  useEffect(() => {
    if (focusRepo === 0) return;
    const id = requestAnimationFrame(() =>
      document.getElementById("dl-repo")?.focus(),
    );
    return () => cancelAnimationFrame(id);
  }, [focusRepo, online]);
  const valid = REPO_RE.test(repo.trim());

  async function submit(body: DownloadRequest): Promise<boolean> {
    setPending(true);
    setError(null);
    setNotice("");
    try {
      const job = await startDownload(connection, body);
      if (job.alreadyPresent)
        setNotice(t("downloads.presentNow", { repo: job.repo }));
      polled.refresh();
      void engine.refresh();
      return true;
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
      return false;
    } finally {
      setPending(false);
    }
  }
  const onSubmit = (ev: React.FormEvent) => {
    ev.preventDefault();
    if (!valid || pending) return;
    const p = parsePatterns(patterns);
    void submit({
      repo: repo.trim(),
      ...(revision.trim() ? { revision: revision.trim() } : {}),
      ...(p.length ? { allow_patterns: p } : {}),
    }).then((ok) => {
      if (ok) setRepo("");
    });
  };
  const resume = (j: DownloadJob) =>
    void submit({
      repo: j.repo,
      ...(j.revision ? { revision: j.revision } : {}),
      ...(j.patterns.length ? { allow_patterns: j.patterns } : {}),
    });
  const cancel = async (j: DownloadJob) => {
    try {
      await cancelDownload(connection, j.id);
      polled.refresh();
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  };
  const shortfall = error ? diskShortfall(error) : null;

  if (polled.unsupported)
    return (
      <DashboardPage data-testid="downloads">
        <PageHeader
          title={t("downloads.title")}
          description={t("downloads.description")}
        />
        <UnavailableNotice
          data-testid="downloads-unsupported"
          title={t("downloads.unsupportedTitle")}
          description={t("downloads.unsupportedDescription")}
        />
      </DashboardPage>
    );
  const free = data?.freeBytes ?? null;
  return (
    <DashboardPage data-testid="downloads">
      <PageHeader
        title={t("downloads.title")}
        description={t("downloads.description")}
      />
      <SectionCard
        icon={Download}
        title={t("downloads.add.title")}
        description={t("downloads.add.description")}
      >
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="grid gap-4 md:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
            <div className="space-y-1.5">
              <Label htmlFor="dl-repo">{t("downloads.add.repo")}</Label>
              <Input
                id="dl-repo"
                value={repo}
                onChange={(e) => setRepo(e.target.value)}
                placeholder="mlx-community/Qwen3-4B-4bit" // i18n-ignore
                autoComplete="off"
                spellCheck={false}
                aria-invalid={repo.length > 0 && !valid}
              />
              {repo.length > 0 && !valid && (
                <p className="text-xs text-error">
                  {t("downloads.add.repoInvalid")}
                </p>
              )}
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="dl-rev">{t("downloads.add.revision")}</Label>
              <Input
                id="dl-rev"
                value={revision}
                onChange={(e) => setRevision(e.target.value)}
                placeholder="main" // i18n-ignore
                autoComplete="off"
                spellCheck={false}
              />
            </div>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="dl-patterns">{t("downloads.add.patterns")}</Label>
            <Input
              id="dl-patterns"
              value={patterns}
              onChange={(e) => setPatterns(e.target.value)}
              placeholder="*.safetensors, *.json" // i18n-ignore
              autoComplete="off"
              spellCheck={false}
            />
            <p className="text-xs text-muted-foreground">
              {t("downloads.add.patternsHint")}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Reasoned
              reason={
                !online
                  ? t("downloads.offlineReason")
                  : !valid
                    ? t("downloads.add.needRepo")
                    : null
              }
            >
              <Button type="submit" disabled={!online || !valid || pending}>
                <Download size={14} />
                {pending
                  ? t("downloads.add.checking")
                  : t("downloads.add.start")}
              </Button>
            </Reasoned>
            <span className="text-xs text-muted-foreground">
              {t("downloads.add.free")} <ByteValue bytes={free} />
            </span>
          </div>
          {shortfall && (
            <div className="space-y-2" data-testid="disk-shortfall">
              <Banner tone="critical" title={t("downloads.disk.title")} />
              <p className="text-sm">
                {t("downloads.disk.body", {
                  needed: bytesText(shortfall.needed),
                  free: bytesText(shortfall.free),
                })}
              </p>
            </div>
          )}
          {notice && (
            <p role="status" className="text-sm text-muted-foreground">
              {notice}
            </p>
          )}
          {error && !shortfall && (
            <div role="alert" className="text-sm text-error">
              <p>{error.message}</p>
              {detailText(error) && (
                <details className="text-xs text-muted-foreground">
                  <summary className="cursor-pointer">
                    {t("downloads.details")}
                  </summary>
                  <p className="mt-1 whitespace-pre-wrap break-words">
                    {detailText(error)}
                  </p>
                </details>
              )}
            </div>
          )}
        </form>
      </SectionCard>
      <Card className="p-2">
        {data && data.jobs.length > 0 ? (
          <div className="space-y-1">
            {data.jobs.map((j) => (
              <JobRow
                key={j.id}
                job={j}
                busy={pending}
                onCancel={(x) => void cancel(x)}
                onResume={resume}
              />
            ))}
          </div>
        ) : (
          <EmptyState
            size="inline"
            icon={<Download size={22} />}
            title={
              data ? t("downloads.emptyTitle") : t("downloads.loadingTitle")
            }
            description={data ? t("downloads.emptyDescription") : undefined}
          />
        )}
      </Card>
      {data?.modelsDir && (
        <p className="text-xs text-muted-foreground">
          {t("downloads.dir")}{" "}
          <span className="break-all font-mono">{data.modelsDir}</span>
        </p>
      )}
      {polled.error && (
        <p role="status" className="text-xs text-muted-foreground">
          {t("downloads.pollError")}
        </p>
      )}
    </DashboardPage>
  );
}
