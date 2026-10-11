import { useEffect, useState } from "react";
import { StatusIndicator } from "@yuhuanowo/yunui";
import { t } from "./i18n/index.ts";
import type { Connection } from "./api.ts";
import {
  getFit,
  UNSUPPORTED,
  type FitResult,
  type Unsupported,
} from "./admin-models-api.ts";
import { ByteValue } from "./ByteValue.tsx";

export const fitStatus = (v: FitResult["verdict"]) =>
  v === "wont_fit" ? "offline" : v === "tight" ? "away" : "online";

export const fitLabel = (v: FitResult["verdict"]) =>
  v === "wont_fit"
    ? t("models.fit.wontFit")
    : v === "tight"
      ? t("models.fit.tight")
      : t("models.fit.fits");

/** Dry-run result of loading one model: the verdict, the arithmetic, and what would be evicted. */
export function FitPanel({ fit }: { fit: FitResult }) {
  const spare =
    fit.freeBytes != null && fit.neededBytes != null
      ? fit.freeBytes - fit.neededBytes
      : null;
  return (
    <div
      className="space-y-3"
      data-testid="fit-panel"
      data-verdict={fit.verdict}
    >
      <StatusIndicator status={fitStatus(fit.verdict)}>
        <span className="text-foreground">{fitLabel(fit.verdict)}</span>
      </StatusIndicator>
      <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1.5 text-sm">
        <dt className="text-muted-foreground">{t("models.fit.needed")}</dt>
        <dd className="text-right">
          <ByteValue bytes={fit.neededBytes} />
        </dd>
        <dt className="pl-3 text-xs text-muted-foreground">
          {t("models.fit.weights")}
        </dt>
        <dd className="text-right text-xs">
          <ByteValue bytes={fit.weightsBytes} />
        </dd>
        <dt className="pl-3 text-xs text-muted-foreground">
          {t("models.fit.kv")}
        </dt>
        <dd className="text-right text-xs">
          <ByteValue bytes={fit.kvReserveBytes} />
        </dd>
        <dt className="text-muted-foreground">{t("models.fit.free")}</dt>
        <dd className="text-right">
          <ByteValue bytes={fit.freeBytes} />
        </dd>
        {spare != null && spare >= 0 && (
          <>
            <dt className="text-muted-foreground">{t("models.fit.spare")}</dt>
            <dd className="text-right">
              <ByteValue bytes={spare} />
            </dd>
          </>
        )}
      </dl>
      {fit.wouldEvict.length > 0 && (
        <div className="text-sm">
          <p className="text-muted-foreground">{t("models.fit.wouldEvict")}</p>
          <ul className="mt-1 space-y-0.5" data-testid="fit-evict">
            {fit.wouldEvict.map((id) => (
              <li key={id} className="break-all font-mono text-xs">
                {id}
              </li>
            ))}
          </ul>
        </div>
      )}
      {fit.estimated && (
        <p className="text-xs text-muted-foreground">
          {t("models.fit.estimated")}
        </p>
      )}
    </div>
  );
}

/** Fetch the fit once per model while `enabled`; "unsupported" hides the feature. */
export function useFit(
  connection: Connection,
  id: string,
  enabled: boolean,
): { fit: FitResult | null; unsupported: boolean; error: string } {
  const [state, setState] = useState<{
    fit: FitResult | null;
    unsupported: boolean;
    error: string;
  }>({ fit: null, unsupported: false, error: "" });
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    setState({ fit: null, unsupported: false, error: "" });
    getFit(connection, id, controller.signal)
      .then((v: FitResult | Unsupported) => {
        if (controller.signal.aborted) return;
        setState(
          v === UNSUPPORTED
            ? { fit: null, unsupported: true, error: "" }
            : { fit: v, unsupported: false, error: "" },
        );
      })
      .catch((e) => {
        if (!controller.signal.aborted)
          setState({
            fit: null,
            unsupported: false,
            error: e instanceof Error ? e.message : String(e),
          });
      });
    return () => controller.abort();
  }, [connection.baseUrl, connection.token, id, enabled]);
  return state;
}
