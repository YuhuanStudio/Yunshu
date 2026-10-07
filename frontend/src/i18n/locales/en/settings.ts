import type zh from "../zh-TW/settings.ts";
import type { Shape } from "../../types.ts";

const settings: Shape<typeof zh> = {
  title: "Settings",
  description: "Connect to the local server and adjust console preferences.",
  "nav.label": "Settings sections",
  "nav.connection": "Engine connection",
  "nav.appearance": "Appearance",
  "nav.models": "Model retention",
  "nav.config": "Effective settings",
  "nav.shortcuts": "Keyboard shortcuts",
  "connection.title": "Engine connection",
  "connection.description": "Which Yunshu server the console talks to.",
  "connection.url": "Server address",
  "connection.urlHelp":
    "Defaults to the same origin. In development Vite forwards to local port 8000.",
  "connection.token": "Access token",
  "connection.tokenHelp":
    "Kept in this page's memory only; enter it again after a reload. Changing the address clears it.",
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
};
export default settings;
