# Authentication, API keys, settings and CORS

Yunshu is a single-node, single-operator server. Access control has three layers, all local.

## Static token

Set `YUNSHU_AUTH_TOKEN` and every request except health, version and docs needs
`Authorization: Bearer <token>` (or `x-api-key`). With no token, inference routes stay open and
operational routes (model load/unload, settings, keys, logs, diagnostics) are denied. The token is an
admin credential. `YUNSHU_AUTH_DISABLED=1` opens everything and is for local development only.

## API keys with quotas

Beyond the static token, the operator can create several keys, each with scopes (`infer`, `admin`),
an optional expiry and quotas (`requests_per_day`, `tokens_per_day`, `max_concurrent`). Manage them in
the console Keys page or with the API (admin scope):

```bash
curl -s -X POST localhost:8000/v1/yunshu/keys -H "Authorization: Bearer $YUNSHU_AUTH_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"name":"laptop","scopes":["infer"],"quotas":{"requests_per_day":1000,"max_concurrent":2}}'
# the response carries "secret" once; only a hash is stored in ~/.yunshu/keys.json (0600)
curl -s localhost:8000/v1/yunshu/usage -H "Authorization: Bearer $YUNSHU_AUTH_TOKEN"
```

`PATCH`, `DELETE` and `POST /v1/yunshu/keys/{id}/rotate` edit, remove and re-issue a key; per-key daily
usage (requests, prompt, completion and cached tokens, errors) is under `GET /v1/yunshu/usage`.
Over a quota the answer is `429` in the route's own error dialect with `Retry-After`. Details:
[API surface](API_SURFACE.md#api-keys-and-quotas).

Limits: keys are not tenants. Authenticated clients share Files, stored completions and Evals data;
there is no per-key isolation or RBAC. The Realtime WebSocket accepts keys but does not count quotas.

## Writing settings

`GET /v1/yunshu/config` lists every setting with value, default, source and when a change applies
(`live`, `reload`, `restart`). `PATCH /v1/yunshu/config` validates and persists changes atomically (all
or nothing, file mode 0600); an environment or CLI value still wins and is reported as `overridden`.
Settings that need a restart are flagged so the console prompts; `POST /v1/yunshu/service/restart`
restarts the server only when it runs under launchd ([Service](SERVICE.md)). `dry_run` previews a change.
The console Settings page does the same with a search box and per-setting controls.

## CORS

`GET`/`PATCH /v1/yunshu/cors` (and the console Settings page) edit the allowed origins live. Origins are
`scheme://host[:port]`, at most 50. A wildcard (`*`) needs an explicit confirmation, is never combined
with other origins and disables credentials: browsers that send cookies or authorization headers
must use explicit `YUNSHU_CORS_ORIGINS`. The same list is checked against the Realtime WebSocket
`Origin` header. Bind address and LAN exposure are read-only here.

## Other protections

- Authorization, API-key and cookie headers are dropped when a model download redirects to another origin.
- Realtime ephemeral secrets (`ek_...`) and credentials in log lines are redacted; transcription secrets
  cannot create model responses.
- `web_fetch` blocks private, loopback and link-local addresses (also after redirects) unless
  `YUNSHU_WEB_FETCH_ALLOW_PRIVATE=1`; repository IDs with traversal are rejected before disk access.
- No usage analytics are sent; see [Telemetry](TELEMETRY.md) for what is sampled locally.
