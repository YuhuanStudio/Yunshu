import type zh from "../zh-TW/common.ts";
import type { Shape } from "../../types.ts";

const common: Shape<typeof zh> = {
  retry: "Retry",
  configure: "Connection settings",
  lastOk: "Last connected at {time}; the last data is kept below.",
  details: "Details",
  copy: "Copy",
  copied: "Copied",
  copyFailed: "Could not write to the clipboard",
  copyLabel: "Copy {label}",
  "language.label": "Language",
  sentenceGap: " ",
  offlineSince: "Engine offline since {time}, for {t}.",
  retryIn: "Retrying in {s} s.",
  retryNow: "Retrying now…",
  engineMessage: "The engine said: {message}",
  retrying:
    "Temporarily unable to update; showing the last reading, retrying automatically",
  staleAt: "Data as of {time}",
  "unlock.placeholder": "Paste the access token",
  "unlock.label": "Unlock key",
  "unlock.remember": "Remember on this device",
  "unlock.submit": "Unlock",
};
export default common;
