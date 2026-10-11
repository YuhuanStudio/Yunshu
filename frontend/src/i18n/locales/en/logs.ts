import type zh from "../zh-TW/logs.ts";
import type { Shape } from "../../types.ts";

const logs: Shape<typeof zh> = {
  "page.title": "Logs",
  "page.description":
    "Recent engine logs, with live tail, search and download.",
  "action.copyVisible": "Copy visible",
  "action.download": "Download visible",
  "action.pause": "Pause",
  "action.resume": "Resume",
  "action.copyLine": "Copy this line",
  "action.jump": "Jump to latest",
  "action.jumpNew": "Jump to latest ({count} new)",
  copiedAll: "Copied {count, plural, one {# line} other {# lines}}",
  copiedLine: "Line copied",
  copyFailed: "Could not write to the clipboard",
  "search.aria": "Search logs",
  "search.placeholder": "Search message or module",
  "level.aria": "Log level",
  "level.all": "All",
  "level.DEBUG": "Debug",
  "level.INFO": "Info",
  "level.WARNING": "Warning",
  "level.ERROR": "Error",
  "level.CRITICAL": "Critical",
  "span.all": "All (in buffer)",
  "span.5m": "Last 5 minutes",
  "span.1h": "Last hour",
  "span.window": "{from}–{to}",
  "live.connecting": "Connecting…",
  "live.live": "Live",
  "live.polling": "Live (polling)",
  "live.reconnecting": "Reconnecting…",
  "live.paused": "Paused",
  "live.window": "Fixed time range",
  "pause.windowWhy":
    "A fixed time range does not follow new records; pick another range to go live again.",
  "list.aria": "Log records",
  "state.loading": "Loading logs…",
  "state.missingTitle": "This engine has no log endpoint yet",
  "state.missingBody":
    "It needs a newer engine version; until then read logs from the terminal or the service log file.",
  "state.deniedTitle": "Admin permission needed",
  "state.deniedBody":
    "Logs are only served to admin keys; check the key this console connects with in Settings.",
  "state.errorTitle": "Logs are not available right now",
  "state.errorBody":
    "The engine did not answer. It retries once the connection is back, or change a filter to try again.",
  "state.emptyTitle": "No logs",
  "state.emptyBody": "Logs the engine writes after it starts appear here.",
  "state.emptyFiltered":
    "Nothing matches the current filters; widen the level, search text or time range.",
  "footer.counts": "{count} lines shown (the server keeps at most {capacity})",
  "footer.dropped": "{count} older lines were pushed out of the buffer",
  "footer.span": "oldest {from}",
  "note.redaction":
    "The server redacts credentials and request content when a line is written, so what you see is already redacted. The buffer lives in memory only (up to {capacity} lines) and is cleared on restart.",
};
export default logs;
