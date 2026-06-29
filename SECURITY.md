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
