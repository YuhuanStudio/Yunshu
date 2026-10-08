# Release readiness: the outward-facing checklist

This page covers what a new user meets before the first token: installing, the first run, running
the server, connecting clients, upgrading, uninstalling, and what happens when something is
wrong. Engine speed and accuracy are tracked in [BENCHMARKS.md](../BENCHMARKS.md).

Source review: 2026-10-08, released v0.1.4; main is the unreleased 0.1.5 cycle.
✅ means the capability exists, with dated runtime evidence stated separately;
🟡 means incomplete and ❌ means missing. This checklist does not replace a
commit-pinned release gate or claim a fresh runtime test.

## Install

| | Status | Notes |
|---|---|---|
| From source (`uv sync --extra vision`) | ✅ | `uv.lock` records exact dependency versions; the vision extra requires mlx-vlm >=0.7.6 |
| `uv tool install` from a wheel | ✅ | Clean tool env on 2026-09-29, then `doctor`, serve Qwen3.5-0.8B, `/health`, `/version`, `/v1/models`, chat, `/v1/messages`, `service install --dry-run` |
| `uv tool install` from git | ✅ | Same package as the wheel; the repository is public |
| PyPI (`uv tool install "yunshu[vision]"`, `pipx install`) | ✅ | v0.1.4 (2026-10-08) released; see [RELEASING.md](../../RELEASING.md) |
| Homebrew | 🟡 P2 | Formula points to the v0.1.3 sdist (repository formula; the external tap is not verified here) in the shared tap [YuhuanStudio/homebrew-tap](https://github.com/YuhuanStudio/homebrew-tap) (`brew install yuhuanstudio/tap/yunshu`) in `packaging/homebrew/`; added after the PyPI release ([steps](../../packaging/homebrew/README.md)) |
| macOS app / menu bar / DMG | — | Not planned for now: CLI + launchd service (decided 2026-09-29) |
| `curl … \| bash` installer with a model suggestion by RAM | ❌ P3 | Rapid-MLX has one; `uv tool install` + `yunshu doctor` covers most of it |
| Python versions | 🟡 P3 | Requires Python >=3.13; CI targets 3.13 |

## First run

| | Status | Notes |
|---|---|---|
| `yunshu doctor` | ✅ | Checks platform / Rosetta, macOS, Python, MLX + Metal, mlx-lm / mlx-vlm, memory, settings, models dir + disk, model exists + fits, port, service. Every problem comes with a fix; exits 1 on failure; `--json` |
| `yunshu pull org/name` | ✅ | Goes to `~/.yunshu/models/org/name`. Refuses when already on disk (models dir or HF cache); resumes an interrupted download; `--force` re-checks |
| `yunshu model list` | ✅ | Models dir (flat or `org/name`) + Hugging Face cache, typed by the server's own detection |
| Models directory | ✅ | Default `~/.yunshu/models`; `yunshu config set models_dir <path>` saves another location in `~/.yunshu/config.toml`; `serve -m org/name` uses a copy in the models dir or the Hugging Face cache; multi-model mode lists both (`YUNSHU_HF_CACHE_MODELS`) |
| `yunshu --version`, grouped `--help` | ✅ | |
| Model picker by RAM / curated list | ❌ P2 | Users still have to know a repo id. A short list of verified models (Qwen3.5 / 3.8 sizes and quantizations) with the memory each needs would close this |

## Serving

| | Status | Notes |
|---|---|---|
| `yunshu serve -m <model>` / `--models-dir` | ✅ | |
| Safe default bind | ✅ | `127.0.0.1`; warns on a network bind without `--auth-token` |
| Stop before loading a missing / half-downloaded / too-large model | ✅ | `serve` runs the doctor model checks |
| Model load fails after the pre-flight | ✅ | The server stays up with `/health/ready` = 503 and a `FATAL` log line (decided 2026-09-29: right for the background service) |
| Background service | ✅ | `yunshu service install/uninstall/start/stop/restart/status/logs` (launchd, restart on crash, `~/Library/Logs/Yunshu`). Plist generation tested; not loaded on the development Mac |
| Health / status endpoints | ✅ | `/health`, `/health/ready` (503 when not ready), `/health/live`, `/version`, `/metrics`; `yunshu status` |
| Settings: one entry point | ✅ | Registry + TOML (`~/.yunshu/config.toml` by default) + `--set` + `yunshu config` / `config set` + generated reference |
| Terminal chat client | ✅ | `yunshu chat`; no admin control plane |
| Service log rotation | ✅ | Size / age rotation, redacted gzip archives, count / age retention; `yunshu service rotate-logs` |

## Clients

| | Status | Notes |
|---|---|---|
| OpenAI SDKs, curl, Anthropic SDK | ✅ | [CLIENTS.md](CLIENTS.md). Auth accepts `Bearer` and `x-api-key` |
| `yunshu launch` for Claude Code / Codex / opencode / Pi | ✅ | |
| Claude Code, Cline, Continue, Open WebUI | 🟡 P2 | Claude Code / Codex / opencode have feature-level E2E evidence in [AGENT_COMPAT.md](AGENT_COMPAT.md); Cline / Continue / Open WebUI remain configuration guidance, not per-release E2E claims |

## Upgrade and uninstall

| | Status | Notes |
|---|---|---|
| Upgrade | 🟡 | `uv tool upgrade yunshu` for the released package; the service needs `yunshu service restart` afterwards ([SERVICE.md](SERVICE.md)) |
| Uninstall | ✅ | Documented in [SERVICE.md](SERVICE.md#uninstalling-yunshu-completely) |

## Release engineering

| | Status | Notes |
|---|---|---|
| One version source | ✅ | `pyproject.toml`; the CLI, `/version`, OpenAPI and MCP read the installed metadata |
| CHANGELOG | ✅ | `## [0.1.4]` entry; version bumped in `pyproject.toml` and `uv.lock` |
| Release workflow | ✅ | Tag-only; checks version / changelog, lint, tests, build, twine, clean install, PyPI behind approval, draft GitHub release |
| CI on push | 🟡 P1 | Lint + build on Linux; macOS unit tests only on release / manual run (to save quota). See the workflow run history; no fresh CI run is claimed by this checklist |
| Third-party notices in the artifacts | ✅ | `THIRD_PARTY_NOTICES.md` ships in the wheel and sdist |
| Telemetry | ✅ | None collected, stated in the README. Outbound connections happen only on user action: model downloads, MCP servers you configure, the configured `web_search` provider, and `web_fetch` for URLs a request names |

## Decisions (2026-09-29)

- The repository is public; v0.1.4 is released. Future releases follow the
  pre-release check.
- CLI + launchd service; no app for now.
- Homebrew through the shared tap `YuhuanStudio/homebrew-tap`.
- Models in `~/.yunshu/models` by default; Hugging Face cache models usable in place; location
  changeable with `yunshu config set models_dir`.
- A failed model load keeps the server up and not ready.
- Docs stay in the README and `docs/`; no telemetry.
