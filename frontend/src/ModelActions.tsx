import { t } from "./i18n/index.ts";
import { useState } from "react";
import { Play, Square, Zap } from "lucide-react";
import {
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@yuhuanowo/yunui";
import { loadModel, warmupModel, type Connection } from "./api";
import { getFit, UNSUPPORTED, type FitResult } from "./admin-models-api";
import { FitPanel } from "./FitPanel";
import { Reasoned } from "./Reasoned";
import { markLoading } from "./loading-clock";
import { ModelManagement } from "./ModelManagement";
import { supportsChat, type Model } from "./ui";
import { fitVerdict } from "./memory-api";

export type Perform = (
  key: string,
  action: () => Promise<unknown>,
) => Promise<void>;

/** Test / warm up / load / unload / manage for one model; shared by the table and the detail view. */
export function ModelActions({
  model,
  connection,
  online,
  busy,
  perform,
  test,
  requestUnload,
  freeGb,
}: {
  model: Model;
  connection: Connection;
  online: boolean;
  busy: string | null;
  perform: Perform;
  test: (id: string) => void;
  requestUnload: (model: Model, button: HTMLButtonElement) => void;
  /** Free GB from the memory ledger; undefined when the server has no ledger. Informs only. */
  freeGb?: number | null;
}) {
  const [checking, setChecking] = useState(false),
    [review, setReview] = useState<FitResult | null>(null);
  const load = () => {
    markLoading(model.id, true);
    void perform(`load:${model.id}`, () => loadModel(connection, model.id));
  };
  /** Dry-run the load first; only a tight or failing verdict stops to ask. */
  async function startLoad() {
    setChecking(true);
    try {
      const v = await getFit(connection, model.id);
      if (v !== UNSUPPORTED && v.verdict !== "fits" && !v.loaded) {
        setReview(v);
        return;
      }
    } catch {
      // the check is advisory: a failed dry run never blocks loading
    } finally {
      setChecking(false);
    }
    load();
  }
  const fit =
    !model.loaded && !model.loading && freeGb !== undefined
      ? fitVerdict(model.size_gb, freeGb)
      : null;
  return (
    <div className="flex flex-wrap items-center gap-1">
      <div className="flex flex-nowrap items-center gap-1 whitespace-nowrap">
        {model.loaded ? (
          <>
            <Button
              variant="secondary"
              size="sm"
              disabled={!online || !supportsChat(model)}
              title={
                supportsChat(model)
                  ? t("models.actions.testTitleChat")
                  : t("models.actions.testTitleApi")
              }
              onClick={() => test(model.id)}
            >
              {t("models.actions.test")}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              disabled={!online || !!busy}
              onClick={() =>
                void perform(`warmup:${model.id}`, () =>
                  warmupModel(connection, { model: model.id, max_tokens: 1 }),
                )
              }
            >
              <Zap size={12} />
              {t("models.actions.warmup")}
            </Button>
          </>
        ) : (
          <Reasoned
            reason={
              !online
                ? t("models.actions.offlineReason")
                : model.loading
                  ? t("models.actions.loadingReason")
                  : busy
                    ? t("models.actions.busyReason")
                    : null
            }
          >
            <Button
              variant="secondary"
              size="sm"
              disabled={!online || !!busy || model.loading || checking}
              title={fit?.text}
              onClick={() => void startLoad()}
            >
              <Play size={12} />
              {checking
                ? t("models.fit.checking")
                : busy?.endsWith(model.id)
                  ? t("models.actions.working")
                  : t("models.actions.load")}
            </Button>
          </Reasoned>
        )}
        {model.loaded && (
          <Reasoned
            reason={
              model.pinned
                ? t("models.actions.pinnedReason")
                : model.loading
                  ? t("models.actions.loadingReason")
                  : !online
                    ? t("models.actions.offlineReason")
                    : busy
                      ? t("models.actions.busyReason")
                      : null
            }
          >
            <Button
              variant="secondary"
              size="sm"
              disabled={!online || !!busy || model.loading || model.pinned}
              onClick={(e) => requestUnload(model, e.currentTarget)}
            >
              <Square size={12} />
              {busy?.endsWith(model.id)
                ? t("models.actions.working")
                : t("models.actions.unload")}
            </Button>
          </Reasoned>
        )}
        <ModelManagement
          connection={connection}
          modelId={model.id}
          disabled={!online || !!busy || model.loading}
          perform={perform}
        />
      </div>
      {/* The hint line keeps its slot whether or not there is a verdict, so polled free memory never moves the row. */}
      {fit && (
        <span
          data-testid={fit.verdict !== "unknown" ? "fit-hint" : undefined}
          data-verdict={fit.verdict !== "unknown" ? fit.verdict : undefined}
          className={`min-h-4 basis-full text-xs ${fit.verdict === "no" ? "text-error" : "text-muted-foreground"}`}
        >
          {fit.verdict !== "unknown" ? fit.text : ""}
        </span>
      )}
      <Dialog open={!!review} onOpenChange={(o) => !o && setReview(null)}>
        <DialogContent closeLabel={t("models.fit.close")}>
          <DialogTitle>
            {t("models.fit.dialogTitle", { id: model.id })}
          </DialogTitle>
          <DialogDescription>
            {review?.verdict === "wont_fit"
              ? t("models.fit.dialogWont")
              : t("models.fit.dialogTight")}
          </DialogDescription>
          {review && <FitPanel fit={review} />}
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={() => setReview(null)}>
              {t("models.fit.cancel")}
            </Button>
            <Reasoned
              reason={
                review?.verdict === "wont_fit"
                  ? t("models.fit.wontReason")
                  : null
              }
            >
              <Button
                disabled={review?.verdict === "wont_fit"}
                onClick={() => {
                  setReview(null);
                  load();
                }}
              >
                {t("models.fit.loadAnyway")}
              </Button>
            </Reasoned>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
