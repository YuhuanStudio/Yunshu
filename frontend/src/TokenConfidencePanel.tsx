import {
  TokenConfidence,
  probability,
  summarizeConfidence,
} from "@yuhuanowo/yunui/patterns";
import { t } from "./i18n/index.ts";
import { number } from "./ui";
import type { TokenLogprob } from "./stream";

/** Token-by-token confidence under a reply, or one honest line when logprobs were asked for and never came. */
export function TokenConfidencePanel({ tokens }: { tokens: TokenLogprob[] }) {
  const sum = summarizeConfidence(tokens);
  return (
    <div className="mt-3" data-testid="token-confidence">
      <TokenConfidence
        tokens={tokens.map((x) => ({
          token: x.token,
          logprob: x.logprob,
          alternatives: x.alternatives,
        }))}
        labels={{
          title: t("playground.conf.title"),
          summary: t("playground.conf.summary", {
            count: number(sum.count, 0),
            mean: number((sum.meanProbability ?? 0) * 100, 0),
            below: number(sum.below50, 0),
          }),
          legend: {
            high: t("playground.conf.high"),
            medium: t("playground.conf.medium"),
            low: t("playground.conf.low"),
            veryLow: t("playground.conf.veryLow"),
          },
          describe: (tok) => {
            const alts = (tok.alternatives ?? [])
              .filter((a) => a.token !== tok.token)
              .slice(0, 3)
              .map(
                (a) => `${a.token} ${number(probability(a.logprob) * 100, 0)}%`,
              )
              .join(t("shell.footer.engine.listSep"));
            return (
              t("playground.conf.token", {
                token: tok.token.trim() || tok.token,
                pct: number(probability(tok.logprob) * 100, 0),
              }) + (alts ? t("playground.conf.alt", { alts }) : "")
            );
          },
          note: t("playground.conf.note"),
        }}
      />
    </div>
  );
}
