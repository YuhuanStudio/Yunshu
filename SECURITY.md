# Security Policy

## Supported versions

Yunshu is pre-1.0 and moves fast. Security fixes land on `main` and in the latest
release; older `0.x` tags are not separately patched.

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for an
unpatched vulnerability.

- Preferred: open a [GitHub private security advisory](https://github.com/YuhuanStudio/Yunshu/security/advisories/new).
- Include: affected version/commit, a minimal reproduction, and the impact.

You'll get an acknowledgement, and we'll coordinate a fix and disclosure timeline
with you. There's no bug-bounty program — this is an open-source project — but
genuine reports are credited.

## Security model — read before exposing the server

Yunshu is designed as a **local, single-consumer** inference server, not a
public-facing multi-tenant service. Its defaults reflect that:

- **Inference endpoints are open by default** (no auth) so a local client works
  out of the box. **Admin endpoints** (model load/unload, profiling, sleep/wake,
  dashboard) are **denied** until `YUNSHU_AUTH_TOKEN` is set.
- `YUNSHU_AUTH_DISABLED=1` removes auth entirely — local convenience only.
- It binds to the host/port you give it and does not sandbox model code.

**If you expose Yunshu beyond localhost**, you are responsible for the perimeter:

- Put it behind a reverse proxy / firewall and require auth (set `YUNSHU_AUTH_TOKEN`).
- Restrict `YUNSHU_CORS_ORIGINS` (defaults to `*`).
- Only load models you trust — `trust_remote_code` paths execute model-supplied code,
  and local-file inputs (image/audio paths) read from the server's filesystem.

Treat an internet-exposed instance without auth as fully open to anyone who can
reach the port.


## Report scope and disclosure

Authentication bypass, unintended file/network access, unsafe model or cache
loading, secret leakage and denial of service through requests are useful report
categories. Include exploit prerequisites (local access, exposed server, trusted
model code), affected version/commit, minimal reproduction and expected impact.
Do not attach real prompts, credentials or private model data. A sanitized
`yunshu diagnose` bundle can help; review it before sharing.

Maintainers assess the affected defaults and supported releases, coordinate a fix
privately, and agree on disclosure and reporter credit. There is no guaranteed
response SLA. A public advisory should identify affected/fixed versions and any
mitigation; do not publish exploit details before coordination.

## Privacy and outbound connections

Yunshu has no usage telemetry or automatic diagnostics upload. Inference runs
locally. Model downloads, remote media URLs, enabled web search/fetch, and configured
MCP servers can make outbound connections; their operators receive the associated
request data. Local inference does not imply every optional tool works offline.

`yunshu diagnose` writes a local redacted bundle without prompts. Optional
`YUNSHU_SERVE_LOG` records numeric serving measurements locally without prompt,
output or token IDs. Prompt caches contain conversation-derived state on disk:
protect the cache directory and diagnostics as local user data. See
[configuration](docs/CONFIGURATION.md) and [API extensions](docs/guides/API_EXTENSIONS.md).

## LAN credentials and stored data

The current server accepts one shared `YUNSHU_AUTH_TOKEN`; it does not provide
per-key isolation or quotas. Authenticated clients share Files, stored chats and
Evals data. Protect these directories like prompts; diagnostics redaction is not
encryption. Configuration writes use owner-only (0600) files. The static token is
stored as plaintext when configured in TOML, because the engine must read it;
protect environment variables and launchd configuration too.

Realtime `ek_` secrets authenticate only Realtime WebSockets until expiry (expiry
limits new connections, not the lifetime of an already accepted session).
Transcription secrets cannot create model responses. Treat these as credentials.
Use explicit CORS origins on LAN; any wildcard disables CORS credentials.
Outbound download headers are bound to the original origin across redirects.
