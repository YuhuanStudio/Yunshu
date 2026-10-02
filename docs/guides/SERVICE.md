# Running Yunshu as a background service

`yunshu service` manages a per-user launchd agent. It starts Yunshu at login, restarts it after a
crash (not after a clean stop), and writes its output to a rotating log file. It does not need root.
The primary files are:

| What | Where |
|---|---|
| launchd agent | `~/Library/LaunchAgents/com.yuhuanstudio.yunshu.plist` |
| log | `~/Library/Logs/Yunshu/yunshu.log` |

## Install

```bash
yunshu doctor -m ~/.yunshu/models/mlx-community/Qwen3.5-9B-MLX-4bit   # optional pre-check
yunshu service install -m ~/.yunshu/models/mlx-community/Qwen3.5-9B-MLX-4bit
```

`install` takes the options `yunshu serve` takes for what to serve and where:

- `--model/-m`: a model folder or a Hugging Face repo id
- `--models-dir/-d`: a directory of models
- `--host`, `--port`: default `127.0.0.1:8000`
- `--config/-c`: a TOML settings file. The service reads this file on every start, so this is the
  place for settings you want to change later.
- `--set KEY=VALUE`: fixed at install time

Two more options:

- `--dry-run` prints the agent and changes nothing.
- `--no-start` writes the agent without starting it.

The service runs `python -m yunshu_cli serve ...` with the Python of the installation that ran
`install`. After upgrading Yunshu in the same environment, run `yunshu service restart`. After
moving to a different environment, run `yunshu service install --force ...` again.

Settings exported in your shell are **not** copied into the service. Put them in the `--config`
file or pass them with `--set`. `HF_HOME`, `HF_HUB_CACHE`, `HF_ENDPOINT` and `HF_HUB_OFFLINE` are
the exception: they are copied, so the service finds the same Hugging Face cache.

A config file looks like this (every name is in [CONFIGURATION.md](../CONFIGURATION.md)):

```toml
# ~/.yunshu/yunshu.toml
YUNSHU_AUTH_TOKEN = "change-me"
YUNSHU_DEFAULT_MAX_TOKENS = 4096
```

## Everyday commands

```bash
yunshu service status        # installed / running (pid) / healthy, plus paths
yunshu service logs -f       # follow the log
yunshu service restart       # e.g. after editing the config file
yunshu service stop          # stop until the next login or `service start`
yunshu service start
yunshu service uninstall     # stop and remove the agent; models and logs stay
```

## Log rotation

`yunshu service rotate-logs` rotates the service log manually. Size / age rotation,
gzip archives and count / age retention use `YUNSHU_LOG_MAX_MB`,
`YUNSHU_LOG_ROTATE_HOURS`, `YUNSHU_LOG_KEEP` and `YUNSHU_LOG_RETENTION_DAYS`.
Secrets are redacted. Main's optional numbers-only serve log is separate; see
[configuration](../CONFIGURATION.md).

## Uninstalling Yunshu completely

```bash
yunshu service uninstall
uv tool uninstall yunshu               # or: pipx uninstall yunshu
rm -rf ~/Library/Logs/Yunshu
rm -rf ~/.yunshu                       # downloaded models: check before deleting
rm -rf ~/.cache/yunshu                 # text-engine SSD prefix cache, if YUNSHU_SSD_CACHE was on
```

The prefix-cache SSD tier lives in `~/.yunshu/cache/apc` (covered by the line above) or in `YUNSHU_VLM_APC_DISK_DIR` if you set it; delete that directory too.
