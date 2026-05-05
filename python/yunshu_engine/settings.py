"""Yunshu Settings — Hierarchical configuration system.

Follows oMLX's settings.py pattern:
- Layering: CLI args > env vars > settings.json > defaults
- System resource auto-detection (RAM, SSD)
- Persistent settings to JSON file
- Adaptive defaults based on available hardware

Usage:
    from yunshu_engine.settings import init_settings, get_settings
    settings = get_settings()
    print(settings.server.port)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SETTINGS_VERSION = "1.0"
DEFAULT_BASE_PATH = Path.home() / ".yunshu"


def get_system_memory() -> int:
    """Return total system RAM in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        pass
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(result.stdout.strip())
    except Exception:
        return 16 * 1024 ** 3


def get_ssd_capacity(path: str | Path) -> int:
    """Return disk capacity in bytes for the given path."""
    path = Path(path).expanduser().resolve()
    check_path = path
    while not check_path.exists() and check_path.parent != check_path:
        check_path = check_path.parent
    try:
        usage = shutil.disk_usage(check_path)
        return usage.total
    except OSError:
        return 500 * 1024 ** 3


def get_gpu_info() -> dict:
    """Detect Apple GPU info."""
    info = {"chip": "Unknown", "gpu_cores": 0, "metal_support": ""}
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "Chipset Model" in line:
                info["chip"] = line.split(":")[-1].strip()
            elif "Total Number of Cores" in line:
                info["gpu_cores"] = int(line.split(":")[-1].strip())
            elif "Metal" in line:
                info["metal_support"] = line.strip()
    except Exception:
        pass
    return info


@dataclass
class ServerSettings:
    """Server configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ServerSettings:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelSettings:
    """Model configuration."""
    model_dirs: list[str] = field(default_factory=list)
    max_model_memory: str = "auto"
    model_fallback: bool = False

    def resolved_model_dirs(self, base_path: Path) -> list[Path]:
        if self.model_dirs:
            return [Path(d).expanduser() for d in self.model_dirs]
        return [base_path / "models"]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ModelSettings:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CacheSettings:
    """KV cache configuration."""
    paged_cache_block_size: int = 256
    max_cache_blocks: Optional[int] = None
    initial_cache_blocks: int = 256
    paged_ssd_cache_dir: Optional[str] = None
    paged_ssd_cache_max_size: str = "100GB"
    hot_cache_max_size: str = "0"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CacheSettings:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EngineSettings:
    """Engine tuning parameters (maps to oMLX SchedulerConfig)."""
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    completion_batch_size: int = 32
    prefill_step_size: int = 2048
    deferred_clear_delay: int = 8
    cache_cleanup_interval: int = 512
    gc_cleanup_interval: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EngineSettings:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class YunshuSettings:
    """Root settings object — follows oMLX's hierarchical pattern."""
    version: str = SETTINGS_VERSION
    server: ServerSettings = field(default_factory=ServerSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    engine: EngineSettings = field(default_factory=EngineSettings)

    # Detected at init
    system_memory_bytes: int = 0
    gpu_info: dict = field(default_factory=dict)
    base_path: str = ""

    def to_dict(self) -> dict:
        d = {
            "version": self.version,
            "server": self.server.to_dict(),
            "models": self.models.to_dict(),
            "cache": self.cache.to_dict(),
            "engine": self.engine.to_dict(),
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> YunshuSettings:
        s = cls()
        if "server" in data:
            s.server = ServerSettings.from_dict(data["server"])
        if "models" in data:
            s.models = ModelSettings.from_dict(data["models"])
        if "cache" in data:
            s.cache = CacheSettings.from_dict(data["cache"])
        if "engine" in data:
            s.engine = EngineSettings.from_dict(data["engine"])
        return s


# ── Singleton ──

_settings: Optional[YunshuSettings] = None


def init_settings(
    base_path: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
) -> YunshuSettings:
    """Initialize settings with hierarchical layering.

    Priority: CLI overrides > env vars > settings.json > defaults
    """
    global _settings

    base = Path(base_path) if base_path else DEFAULT_BASE_PATH
    base.mkdir(parents=True, exist_ok=True)

    settings_file = base / "settings.json"

    # 1. Start with defaults
    _settings = YunshuSettings()
    _settings.base_path = str(base)

    # 2. Load settings.json if it exists
    if settings_file.exists():
        try:
            with open(settings_file) as f:
                file_data = json.load(f)
            _settings = YunshuSettings.from_dict(file_data)
            _settings.base_path = str(base)
            logger.info(f"Loaded settings from {settings_file}")
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")

    # 3. Apply env vars
    _apply_env_vars(_settings)

    # 4. Apply CLI overrides
    if cli_overrides:
        _apply_overrides(_settings, cli_overrides)

    # 5. Detect system resources
    _settings.system_memory_bytes = get_system_memory()
    _settings.gpu_info = get_gpu_info()

    # 6. Save for next time
    try:
        with open(settings_file, "w") as f:
            json.dump(_settings.to_dict(), f, indent=2)
    except Exception:
        pass

    logger.info(
        f"Settings initialized: "
        f"mem={_settings.system_memory_bytes / 1024**3:.0f}GB, "
        f"gpu={_settings.gpu_info.get('chip', 'Unknown')}, "
        f"port={_settings.server.port}"
    )
    return _settings


def get_settings() -> YunshuSettings:
    """Get current settings (initialize if needed)."""
    global _settings
    if _settings is None:
        return init_settings()
    return _settings


def save_settings() -> None:
    """Persist current settings to disk."""
    if _settings is None:
        return
    settings_file = Path(_settings.base_path) / "settings.json"
    try:
        with open(settings_file, "w") as f:
            json.dump(_settings.to_dict(), f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save settings: {e}")


def _apply_env_vars(s: YunshuSettings) -> None:
    """Apply environment variable overrides."""
    if v := os.environ.get("YUNSHU_HOST"):
        s.server.host = v
    if v := os.environ.get("YUNSHU_PORT"):
        s.server.port = int(v)
    if v := os.environ.get("YUNSHU_LOG_LEVEL"):
        s.server.log_level = v
    if v := os.environ.get("YUNSHU_MODEL"):
        s.models.model_dirs = [v]
    if v := os.environ.get("YUNSHU_MODELS_DIR"):
        s.models.model_dirs = [v]
    if v := os.environ.get("YUNSHU_MAX_MEMORY_GB"):
        s.models.max_model_memory = f"{v}GB"


def _apply_overrides(s: YunshuSettings, overrides: dict) -> None:
    """Apply CLI overrides to settings."""
    for key, value in overrides.items():
        if value is None:
            continue
        parts = key.split(".", 1)
        if len(parts) == 2:
            group, attr = parts
            target = getattr(s, group, None)
            if target and hasattr(target, attr):
                setattr(target, attr, value)
        elif hasattr(s, key):
            setattr(s, key, value)
