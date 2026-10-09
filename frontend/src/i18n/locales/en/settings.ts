import type zh from "../zh-TW/settings.ts";
import type { Shape } from "../../types.ts";

const settings: Shape<typeof zh> = {
  title: "Settings",
  description: "Connect to the local server and adjust console preferences.",
  "nav.label": "Settings sections",
  "nav.connection": "Engine connection",
  "nav.jump": "Jump to section",
  "nav.appearance": "Appearance",
  "nav.models": "Model retention",
  "nav.config": "Effective settings",
  "nav.service": "Service",
  "nav.network": "Network and CORS",
  "nav.shortcuts": "Keyboard shortcuts",
  "connection.title": "Engine connection",
  "connection.description": "Which Yunshu server the console talks to.",
  "connection.url": "Server address",
  "connection.urlHelp":
    "Defaults to the same origin; a local server listens on port 8000 by default. In development Vite forwards to the configured engine address.",
  "connection.token": "Access token",
  "connection.tokenHelp":
    "By default kept in this page's memory only; enter it again after a reload. Changing the address clears it.",
  "connection.remember": "Remember the token on this device",
  "connection.rememberHelp":
    "Stores the token as plain text in this browser's localStorage so you need not enter it after a reload. Keep it off on a shared computer; turning it off deletes the stored token at once.",
  "connection.rememberFailed":
    "The browser would not store it; the token stays in this page's memory only.",
  "connection.tokenPlaceholder":
    "Leave empty when the server has no authentication",
  "connection.showToken": "Show token",
  "connection.hideToken": "Hide token",
  "connection.save": "Save and connect",
  "connection.invalid":
    "Use an HTTP(S) server address without credentials, query or fragment.",
  "connection.invalidShort": "Invalid server address",
  "appearance.title": "Appearance",
  "appearance.dark": "Dark mode",
  "appearance.darkHelp": "Saved in this browser.",
  "appearance.language": "Language",
  "appearance.languageHelp": "Saved in this browser; applies immediately.",
  "permissions.title": "Model action permissions",
  "permissions.description":
    "Loading and unloading need permission from the server. On a 401, use the YUNSHU_AUTH_TOKEN the server was started with. This page does not change engine launch options or turn authentication off.",
  "shortcuts.title": "Keyboard shortcuts",
  "shortcuts.palette": "Open the command palette to jump to a page or model",
  "shortcuts.send": "Send a message in the playground",
  "shortcuts.close": "Close dialogs and menus",
  "admin.cancel": "Cancel",
  "admin.dismiss": "Dismiss",
  "admin.error.network":
    "Could not reach the engine. Check that the server is running.",
  "admin.error.denied":
    "An admin token is required. Enter the server's YUNSHU_AUTH_TOKEN under Engine connection.",
  "admin.error.unsupported":
    "This engine version does not support this action.",
  "admin.error.conflict": "This cannot be done in the current state.",
  "admin.error.invalid": "The server rejected these values.",
  "admin.error.server": "The engine answered {status}. Try again shortly.",
  "admin.error.shape": "The engine returned something unrecognised.",
  "config.applies": "Applies",
  "config.applies.live": "Now",
  "config.applies.reload": "Next load",
  "config.applies.restart": "Restart",
  "config.pending":
    "{count, plural, one {# unsaved change} other {# unsaved changes}}",
  "config.noChanges": "No unsaved changes",
  "config.discard": "Discard changes",
  "config.preview": "Preview",
  "config.previewing": "Previewing…",
  "config.save": "Save",
  "config.saving": "Saving…",
  "config.fixFirst": "Fix the fields marked with errors first",
  "config.reset": "Reset to default",
  "config.resetAria": "Reset {name} to its default",
  "config.willReset": "Resets to the default when saved",
  "config.increment": "Increase",
  "config.decrement": "Decrease",
  "config.secretSet": "Set (enter a new value to replace)",
  "config.secretUnset": "Not set",
  "config.overriddenEnv":
    "Environment variable {name} is set and wins over this value.",
  "config.overriddenCli":
    "A command-line flag is set and wins over this value.",
  "config.invalidCount":
    "{count, plural, one {# setting failed validation} other {# settings failed validation}}; nothing was changed",
  "config.unsupported":
    "This engine version cannot change settings from the console yet",
  "config.unsupportedHelp":
    "Use yunshu config set, or edit the config file and restart.",
  "config.experimentalTitle": "Change experimental settings",
  "config.experimentalBody":
    "Experimental settings are temporary: once measured they may be removed or made the default, and their behaviour may not be stable. Change these settings anyway?",
  "config.experimentalConfirm": "Change them",
  "config.sum.title": "Saved",
  "config.sum.previewTitle": "Preview (not saved)",
  "config.sum.applied": "{count} applied now",
  "config.sum.reload": "{count} take effect after a model reload",
  "config.sum.restart": "{count} take effect after a restart",
  "config.sum.overridden":
    "{count} overridden by the environment or command line",
  "config.sum.reloadHelp": "These apply once the model is reloaded.",
  "config.sum.restartHelp": "These apply once the engine restarts.",
  "config.reloadModel": "Reload model",
  "config.reloaded": "Model reloaded",
  "config.reloadFailed": "Reload failed. Unload and load it again from Models.",
  "config.manualRestart": "Command to restart by hand",
};
export default settings;
