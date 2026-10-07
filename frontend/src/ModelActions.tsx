import { Play, Square, Zap } from "lucide-react";
import { Button } from "@yuhuanowo/yunui";
import { loadModel, warmupModel, type Connection } from "./api";
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
  const fit =
    !model.loaded && !model.loading && freeGb !== undefined
      ? fitVerdict(model.size_gb, freeGb)
      : null;
  return (
    <div className="flex flex-wrap items-center gap-1">
      {model.loaded ? (
        <>
          <Button
            size="sm"
            disabled={!online || !supportsChat(model)}
            title={
              supportsChat(model)
                ? "文字或視覺推理測試"
                : "此模型請使用 API 接入對應端點"
            }
            onClick={() => test(model.id)}
          >
            測試
          </Button>
          <Button
            variant="ghost"
            size="sm"
            disabled={!online || !!busy}
            onClick={() =>
              void perform(`warmup:${model.id}`, () =>
                warmupModel(connection, { model: model.id, max_tokens: 1 }),
              )
            }
          >
            <Zap size={12} />
            預熱
          </Button>
        </>
      ) : (
        <Button
          size="sm"
          disabled={!online || !!busy || model.loading}
          title={fit?.text}
          onClick={() =>
            void perform(`load:${model.id}`, () =>
              loadModel(connection, model.id),
            )
          }
        >
          <Play size={12} />
          {busy?.endsWith(model.id) ? "處理中" : "載入"}
        </Button>
      )}
      {model.loaded && (
        <Button
          variant="secondary"
          size="sm"
          disabled={!online || !!busy || model.loading || model.pinned}
          onClick={(e) => requestUnload(model, e.currentTarget)}
        >
          <Square size={12} />
          {busy?.endsWith(model.id) ? "處理中" : "卸載"}
        </Button>
      )}
      {fit && fit.verdict !== "unknown" && (
        <span
          data-testid="fit-hint"
          data-verdict={fit.verdict}
          className={`basis-full text-xs ${fit.verdict === "no" ? "text-error" : "text-muted-foreground"}`}
        >
          {fit.text}
        </span>
      )}
      <ModelManagement
        connection={connection}
        modelId={model.id}
        disabled={!online || !!busy || model.loading}
        perform={perform}
      />
    </div>
  );
}
