import type zh from "../zh-TW/common.ts";
import type { Shape } from "../../types.ts";

const common: Shape<typeof zh> = {
  retry: "Retry",
  configure: "Connection settings",
  autoRetry: "Retrying every 3 seconds.",
  lastOk: "Last connected at {time}; the last data is kept below.",
  details: "Details",
  copy: "Copy",
  copied: "Copied",
  copyFailed: "Could not write to the clipboard",
  copyLabel: "Copy {label}",
  "language.label": "Language",
};
export default common;
