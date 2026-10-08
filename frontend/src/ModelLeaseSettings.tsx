import { t } from "./i18n/index.ts";
import { useEffect, useState } from "react";
import {
  Button,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@yuhuanowo/yunui";
import { SettingRow } from "@yuhuanowo/yunui/patterns";
import { warmupModel, type Connection } from "./api";
import { MemoryStick } from "lucide-react";
import { SectionCard, modelLabel, type Engine } from "./ui";
import type { Perform } from "./Models";
export function ModelLeaseSettings({
  connection,
  engine,
  perform,
  busy,
}: {
  connection: Connection;
  engine: Engine;
  perform: Perform;
  busy: boolean;
}) {
  const eligible = (engine.status?.models ?? []).filter(
    (model) => !model.pinned,
  );
  const [model, setModel] = useState(""),
    [keepAlive, setKeepAlive] = useState("5m");
  useEffect(() => {
    if (!eligible.some((item) => item.id === model))
      setModel(eligible[0]?.id ?? "");
  }, [engine.status?.models, model]);
  return (
    <SectionCard
      icon={MemoryStick}
      title={t("models.lease.title")}
      description={t("models.lease.description")}
      bodyClassName="px-4 pb-4"
    >
      <SettingRow
        divider={false}
        title={t("models.lease.model")}
        description={t("models.lease.modelHint")}
        control={
          <Select value={model || undefined} onValueChange={setModel}>
            <SelectTrigger
              aria-label={t("models.lease.modelAria")}
              className="w-full sm:w-64"
            >
              <SelectValue placeholder={t("models.lease.modelPlaceholder")} />
            </SelectTrigger>
            <SelectContent>
              {eligible.map((item) => (
                <SelectItem key={item.id} value={item.id}>
                  {modelLabel(item.id)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        }
      />
      <SettingRow
        divider={false}
        title={t("models.lease.idle")}
        description={t("models.lease.idleHint")}
        control={
          <Select value={keepAlive} onValueChange={setKeepAlive}>
            <SelectTrigger aria-label={t("models.lease.idle")} className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[
                ["5m", t("models.lease.5m")],
                ["15m", t("models.lease.15m")],
                ["1h", t("models.lease.1h")],
                ["-1", t("models.lease.forever")],
              ].map(([value, label]) => (
                <SelectItem key={value} value={value}>
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        }
      />
      <div className="pt-4">
        <Button
          disabled={!model || busy || engine.phase !== "online"}
          onClick={() =>
            void perform(`warmup:${model}`, () =>
              warmupModel(connection, {
                model,
                keep_alive: keepAlive === "-1" ? -1 : keepAlive,
                max_tokens: 1,
              }),
            )
          }
        >
          {t("models.lease.apply")}
        </Button>
      </div>
      {!eligible.length && (
        <p className="text-xs text-muted-foreground">
          {t("models.lease.none")}
        </p>
      )}
    </SectionCard>
  );
}
