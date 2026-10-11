import type zh from "../zh-TW/service.ts";
import type { Shape } from "../../types.ts";

const service: Shape<typeof zh> = {
  title: "Service",
  description:
    "Status of the engine as a background service (launchd), and restart.",
  refresh: "Refresh",
  loading: "Loading service status…",
  "unavailable.unsupported.title": "This engine has no service management",
  "unavailable.unsupported.description":
    "Upgrade Yunshu to see status and restart from here.",
  "unavailable.denied.title": "Admin token required",
  "unavailable.denied.description":
    "Enter the server's YUNSHU_AUTH_TOKEN, or a key with admin scope, under Engine connection.",
  "unavailable.error.title": "Could not load",
  "unavailable.error.description":
    "Check that the server is running, then try again.",
  "state.managed": "Managed by launchd",
  "state.other": "Loaded in launchd, but this process was not started by it",
  "state.stopped": "Installed, not running",
  "state.notInstalled": "Not installed as a service",
  "row.status": "Status",
  "row.pid": "Process ID",
  "row.uptime": "Running for",
  "row.uptimeHelp": "Counted from this process's start.",
  "row.version": "Version",
  "row.plist": "plist file",
  "row.log": "Log file",
  "restart.button": "Restart",
  "restart.confirmTitle": "Restart the engine",
  "restart.confirmBody":
    "In-flight requests get up to {seconds} s to finish, then the engine is restarted. The server is unreachable while it restarts and loaded models must load again.",
  "restart.confirmBodyUnknown":
    "In-flight requests get time to finish, then the engine is restarted. The server is unreachable while it restarts and loaded models must load again.",
  "restart.confirm": "Restart",
  "restart.accepted":
    "Restart scheduled: waiting for {count, plural, one {# in-flight request} other {# in-flight requests}} (up to {seconds} s). The console reconnects on its own.",
  "restart.failed":
    "The restart request did not go through. Try again shortly.",
  "restart.notLaunchd":
    "This engine was not started by the launchd service, so it cannot restart itself. Restart it by hand:",
  "restart.manual": "Command to restart by hand",
  "restart.cli": "Command",
  "restart.help":
    "One-click restart works only when launchd manages the engine.",
  "network.title": "Network",
  "network.description":
    "The address the server listens on (read-only). To change it, reinstall the service.",
  "network.address": "Address the console uses",
  "network.addressHelp": "From the current server address setting.",
  "network.exposure": "Reachable from",
  "network.exposureHelp":
    "This machine only means no other computer can connect; local network means other devices on it can.",
  "network.loopback": "This machine only",
  "network.lan": "Local network",
  "network.cmdLocal": "Switch to this machine only",
  "network.cmdLan": "Switch to the local network",
  "network.lanWarning":
    "Before opening the local network, set YUNSHU_AUTH_TOKEN or create an API key, and allow only the origins you need in CORS.",
  "cors.title": "CORS origins",
  "cors.description":
    "Which websites may call this server from a browser. Changes apply immediately.",
  "cors.invalid":
    "Enter an origin starting with http:// or https:// and no path, such as https://app.example.com, or a lone *.",
  "cors.mixed": "* cannot be combined with other origins.",
  "cors.duplicate": "This origin is already in the list.",
  "cors.rejected": "The server rejected these origins: {list}",
  "cors.forced":
    "Currently set by the {source}, which wins over the config file. Edits here are saved but do not change the running value.",
  "cors.source.env": "environment variable YUNSHU_CORS_ORIGINS",
  "cors.source.cli": "command-line flag",
  "cors.wildcardTitle": "Any website can call this server",
  "cors.wildcardBody":
    "* lets a page on any website send requests to this server, without credentials. Unless the server is reachable only from this machine, use explicit origins instead.",
  "cors.thisAllowed": "This console's origin {origin} is allowed.",
  "cors.thisBlocked":
    "This console's origin {origin} is not in the list; pages from other origins are blocked.",
  "cors.listAria": "Allowed origins",
  "cors.remove": "Remove {origin}",
  "cors.none": "The list is empty",
  "cors.add": "Add an origin",
  "cors.addButton": "Add",
  "cors.confirmAny": "I understand * lets any website call this server",
  "cors.confirmAnyNeeded": "Confirm before using *",
  "cors.save": "Save",
  "cors.reset": "Reset to default",
  "cors.alreadyDefault": "Already the default",
  "cors.saved": "Saved",
  "cors.help": "Default: {default}.",
};
export default service;
