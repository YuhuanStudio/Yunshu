import { useEffect, useState } from "react";
import {
  Button,
  Card,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@yuhuanowo/yunui";
import { SettingRow } from "@yuhuanowo/yunui/patterns";
import { warmupModel, type Connection } from "./api";
import { modelLabel, type Engine } from "./ui";
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
    <Card className="px-5 pb-5">
      <h2 className="pt-5 text-sm font-semibold">模型保留時間</h2>
      <SettingRow
        title="模型"
        description="固定保留的單模型不適用。"
        control={
          <Select value={model || undefined} onValueChange={setModel}>
            <SelectTrigger
              aria-label="保留時間的模型"
              className="w-full sm:w-64"
            >
              <SelectValue placeholder="選擇非固定模型" />
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
        title="閒置保留時間"
        description="透過模型預熱介面更新保留時間，並執行一次短預熱。"
        control={
          <Select value={keepAlive} onValueChange={setKeepAlive}>
            <SelectTrigger aria-label="閒置保留時間" className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[
                ["5m", "5 分鐘"],
                ["15m", "15 分鐘"],
                ["1h", "1 小時"],
                ["-1", "持續保留"],
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
          套用並預熱
        </Button>
      </div>
      {!eligible.length && (
        <p className="text-xs text-muted-foreground">
          目前沒有可調整的模型。需由多模型服務註冊非固定模型。
        </p>
      )}
    </Card>
  );
}
