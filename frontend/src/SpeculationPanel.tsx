import { useMemo } from "react";
import { Table, Tbody, Td, Th, Thead, Tr } from "@yuhuanowo/yunui";
import { ChartCard } from "./AnalyticsPanels";
import { t, useLocale } from "./i18n/index.ts";
import type { Row } from "./RequestTrace";
import { summarizeSpeculation } from "./speculation";
import { number } from "./ui";

/**
 * What speculation actually did for the engine's recent finished requests: the mode each request
 * reported, token-weighted acceptance with its drafted/accepted denominators, and rounds. Nothing
 * is inferred from configuration, and per-depth acceptance is stated as not reported.
 */
export function SpeculationPanel({ rows }: { rows: readonly Row[] }) {
  useLocale();
  const s = useMemo(() => summarizeSpeculation(rows), [rows]);
  if (s.total === 0) return null;
  return (
    <ChartCard
      data-testid="speculation-panel"
      title={t("requests.spec.title")}
      description={t("requests.spec.desc", {
        n: number(s.total, 0),
        engaged: number(s.engaged, 0),
      })}
    >
      {s.modes.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {t("requests.spec.none")}
        </p>
      ) : (
        <Table scrollLabel={t("requests.spec.title")}>
          <Thead>
            <Tr>
              <Th>{t("requests.spec.col.mode")}</Th>
              <Th>{t("requests.spec.col.requests")}</Th>
              <Th title={t("requests.spec.acceptHelp")}>
                {t("requests.spec.col.accept")}
              </Th>
              <Th>{t("requests.spec.col.tokens")}</Th>
              <Th>{t("requests.spec.col.rounds")}</Th>
            </Tr>
          </Thead>
          <Tbody>
            {s.modes.map((m) => (
              <Tr key={m.mode} data-mode={m.mode}>
                <Td className="font-mono text-xs">{m.mode}</Td>
                <Td className="tabular-nums">{number(m.requests, 0)}</Td>
                <Td
                  className="whitespace-nowrap tabular-nums"
                  title={t("requests.spec.acceptHelp")}
                >
                  {m.acceptance == null
                    ? "—"
                    : `${number(m.acceptance * 100, 1)}%`}
                </Td>
                <Td className="whitespace-nowrap tabular-nums">
                  {m.counted === 0
                    ? "—"
                    : `${number(m.accepted, 0)} / ${number(m.drafted, 0)}`}
                </Td>
                <Td
                  className="whitespace-nowrap tabular-nums"
                  title={
                    m.copyRounds > 0
                      ? t("requests.spec.copy", {
                          rounds: number(m.copyRounds, 0),
                          tokens: number(m.copyTokens, 0),
                        })
                      : undefined
                  }
                >
                  {m.roundsReported === 0 ? "—" : number(m.rounds, 0)}
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
      )}
      <div className="mt-3 space-y-1 text-xs leading-5 text-muted-foreground">
        {s.plain > 0 && s.engaged > 0 && (
          <p>{t("requests.spec.plain", { n: number(s.plain, 0) })}</p>
        )}
        {s.unattributed > 0 && (
          <p>
            {t("requests.spec.unattributed", { n: number(s.unattributed, 0) })}
          </p>
        )}
        <p>{t("requests.spec.depth")}</p>
      </div>
    </ChartCard>
  );
}
