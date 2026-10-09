import type zh from "../zh-TW/keys.ts";
import type { Shape } from "../../types.ts";

const keys: Shape<typeof zh> = {
  title: "API keys",
  description:
    "Give each app or person its own key, with its own scopes, quotas and expiry.",
  create: "Create key",
  loading: "Loading keys…",
  "unavailable.unsupported.title": "This engine has no key management",
  "unavailable.unsupported.description":
    "Upgrade Yunshu. Until then, the YUNSHU_AUTH_TOKEN set at startup still works.",
  "unavailable.denied.title": "Admin token required",
  "unavailable.denied.description":
    "Key management is for admins. Enter the server's YUNSHU_AUTH_TOKEN, or a key with admin scope, in Settings.",
  "unavailable.error.title": "Could not load keys",
  "unavailable.error.description":
    "Check that the server is running, then try again.",
  "secret.titleCreate": 'Key "{name}" created',
  "secret.titleRotate": 'Key "{name}" rotated',
  "secret.warning":
    "This is the only time the full key is shown. You cannot view it again after leaving this page, so copy and store it now. After a rotation the old key stops working immediately.",
  "secret.label": "Full key",
  "secret.use":
    "Clients send it as Authorization: Bearer, or set it as the {env} environment variable.",
  "secret.done": "I have saved it",
  "list.title": "Keys",
  "list.description":
    "{count, plural, one {# key} other {# keys}}. Usage covers the last 24 hours, the same window the quotas use.",
  "empty.title": "No keys yet",
  "empty.description":
    "Once you create a key the server starts requiring authentication; YUNSHU_AUTH_TOKEN stays an admin key.",
  "table.aria": "API keys",
  "table.name": "Name",
  "table.scopes": "Scopes",
  "table.requests": "Requests, last 24 h",
  "table.tokens": "Tokens, last 24 h",
  "table.lastUsed": "Last used",
  "table.expires": "Expires",
  "table.enabled": "Enabled",
  "table.actions": "Actions",
  unlimited: "no limit",
  never: "Never",
  expired: "Expired",
  toggleAria: "Enable or disable key {name}",
  edit: "Edit",
  "rotate.button": "Rotate",
  "rotate.title": "Rotate key",
  "rotate.body":
    'A new key is generated for "{name}" and the old one stops working at once. Apps using the old key must switch.',
  "rotate.confirm": "Rotate",
  "delete.button": "Delete",
  "delete.title": "Delete key",
  "delete.body":
    'After "{name}" is deleted, apps using it get 401 immediately. This cannot be undone.',
  "delete.confirm": "Delete",
  "scope.infer": "Inference",
  "scope.admin": "Admin",
  "scope.inferHelp": "Call models: chat, completions, embeddings and so on.",
  "scope.adminHelp": "Manage models, settings, keys and the service.",
  "quota.requests_per_day": "Requests per day",
  "quota.tokens_per_day": "Tokens per day",
  "quota.max_concurrent": "Concurrent requests",
  "form.createTitle": "Create key",
  "form.editTitle": "Edit key",
  "form.create": "Create",
  "form.save": "Save",
  "form.name": "Name",
  "form.namePlaceholder": "For example: Claude Code on my laptop",
  "form.nameRequired": "Enter a name",
  "form.scopeRequired": "Choose at least one scope",
  "form.scopes": "Scopes",
  "form.adminWarning":
    "Admin scope can change settings and other keys. Give it only to trusted uses.",
  "form.quotaHelp": "0 means no limit. Daily limits count the last 24 hours.",
  "form.expires": "Expires (leave empty for never)",
  "usage.title": "Usage",
  "usage.description": "Counted per day (UTC); days without records show 0.",
  "usage.keyAria": "Choose a key",
  "usage.allKeys": "All keys",
  "usage.days": "{count} days",
  "usage.empty": "No usage recorded in this period.",
  "usage.requests": "Requests",
  "usage.tokens": "Input plus output tokens",
};
export default keys;
