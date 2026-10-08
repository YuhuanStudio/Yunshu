import type { EngineStatus, RequestRow } from "./api.ts";

/**
 * What unloading (or replacing) a model does to work in flight, from the engine status only.
 * The engine reports which requests are running and, per row, the model id; it does not say
 * whether a request could finish first, so the preview lists what WOULD be interrupted and never
 * promises a graceful drain.
 */
export interface UnloadImpact {
  /** Requests known to run on this model. */
  rows: RequestRow[];
  /** Requests whose model the engine did not report, while this model is one of several loaded. */
  unattributed: number;
}

export function unloadImpact(
  status: EngineStatus | null,
  modelId: string,
): UnloadImpact {
  const items = status?.requests.items ?? [];
  const loaded = (status?.models ?? []).filter((m) => m.loaded);
  const onlyOne = loaded.length === 1 && loaded[0].id === modelId;
  const rows: RequestRow[] = [];
  let unattributed = 0;
  for (const row of items) {
    if (row.cancelled) continue;
    if (row.model) {
      if (row.model === modelId) rows.push(row);
    } else if (onlyOne) rows.push(row);
    else unattributed += 1;
  }
  return { rows, unattributed };
}
