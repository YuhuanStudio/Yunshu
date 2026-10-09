import { t } from "./i18n/index.ts";
import { Button, Card, StatusIndicator } from "@yuhuanowo/yunui";
import { SectionRow } from "@yuhuanowo/yunui/patterns";
import { HardDrive, Play } from "lucide-react";
import { loadModel, type Connection } from "./api";
import { isUnsupported, type LocalModel } from "./admin-models-api";
import { ByteValue } from "./ByteValue";
import { ModelManagement } from "./ModelManagement";
import { Reasoned } from "./Reasoned";
import type { Perform } from "./ModelActions";
import { number } from "./ui";

/** Human facts about one on-disk model: quant, context, capabilities. Unknown parts are left out, not zeroed. */
export function localFacts(m: LocalModel): string[] {
  const out: string[] = [];
  if (m.quantBits != null)
    out.push(t("models.local.quant", { bits: m.quantBits }));
  if (m.contextLength != null)
    out.push(t("models.local.context", { tokens: number(m.contextLength, 0) }));
  if (m.parameters) out.push(m.parameters);
  if (m.modelType) out.push(m.modelType);
  return out;
}

const CAPS = ["vision", "audio", "tools", "reasoning", "embedding"] as const;
export const capabilityLabel = (c: string) => {
  switch (c) {
    case "vision":
      return t("models.local.cap.vision");
    case "audio":
      return t("models.local.cap.audio");
    case "tools":
      return t("models.local.cap.tools");
    case "reasoning":
      return t("models.local.cap.reasoning");
    case "embedding":
      return t("models.local.cap.embedding");
    default:
      return c;
  }
};
const knownCaps = (m: LocalModel) =>
  m.capabilities.filter((c) => (CAPS as readonly string[]).includes(c));

/** On-disk models the engine has not registered, with the actions that make them usable. */
export function LocalInventory({
  items,
  connection,
  online,
  busy,
  perform,
  freeBytes,
}: {
  items: LocalModel[];
  connection: Connection;
  online: boolean;
  busy: string | null;
  perform: Perform;
  freeBytes: number | null;
}) {
  if (!items.length) return null;
  const register = (m: LocalModel) => {
    void perform(`load:${m.id}`, async () => {
      try {
        return await loadModel(connection, m.id);
      } catch (e) {
        // a server that cannot register a folder says 404: explain, do not dump it
        if (isUnsupported(e))
          throw new Error(t("models.local.cannotRegister", { id: m.id }));
        throw e;
      }
    });
  };
  return (
    <div className="space-y-3" data-testid="local-inventory">
      <SectionRow
        title={
          <span className="flex items-center gap-2">
            {t("models.local.title")}
            <span className="tabular-nums text-muted-foreground">
              {items.length}
            </span>
          </span>
        }
      />
      <Card className="divide-y-0 p-2">
        {items.map((m) => {
          const caps = knownCaps(m);
          const reason = !online
            ? t("models.actions.offlineReason")
            : busy
              ? t("models.actions.busyReason")
              : !m.complete
                ? t("models.local.incompleteReason")
                : null;
          return (
            <div
              key={m.path}
              data-testid="local-row"
              className="flex flex-wrap items-start gap-x-4 gap-y-2 rounded-md px-3 py-3"
            >
              <HardDrive
                size={16}
                className="mt-0.5 shrink-0 text-muted-foreground"
              />
              <div className="min-w-0 flex-1 basis-60">
                <p className="break-all text-sm font-medium">{m.id}</p>
                <p className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                  <ByteValue bytes={m.sizeBytes} />
                  {localFacts(m).map((f) => (
                    <span key={f}>{f}</span>
                  ))}
                  {caps.length > 0 && (
                    <span>{caps.map(capabilityLabel).join(" · ")}</span>
                  )}
                  <StatusIndicator status={m.complete ? "online" : "away"}>
                    {m.complete
                      ? t("models.local.complete")
                      : t("models.local.incomplete")}
                  </StatusIndicator>
                </p>
                {!m.complete && m.completeReason && (
                  <p className="mt-1 break-words text-xs text-muted-foreground">
                    {m.completeReason}
                  </p>
                )}
                <p className="mt-1 break-all font-mono text-xs text-muted-foreground">
                  {m.path}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-1">
                <Reasoned reason={reason}>
                  <Button
                    size="sm"
                    disabled={!!reason}
                    onClick={() => register(m)}
                  >
                    <Play size={12} />
                    {t("models.local.register")}
                  </Button>
                </Reasoned>
                <ModelManagement
                  connection={connection}
                  modelId={m.id}
                  disabled={!online || !!busy}
                  perform={perform}
                />
              </div>
            </div>
          );
        })}
      </Card>
      {freeBytes != null && (
        <p className="text-xs text-muted-foreground">
          {t("models.local.free")} <ByteValue bytes={freeBytes} />
        </p>
      )}
    </div>
  );
}
