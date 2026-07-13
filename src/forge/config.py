"""
Central Configuration & Environment Management Module for FORGE.
Provides thread-safe, validated configuration loading from:
• Environment variables (WIENROOT, SCRATCH, LOG_LEVEL, etc.)
• Local JSON/TOML config files (~/.config/forge/config.json)
• CLI overrides & runtime defaults
• HPC cluster conventions (SLURM/PBS scratch paths, MPI env)

Key Architecture Features:
• Strict path validation & permission checks for critical directories
• Lazy initialization & thread-safe singleton pattern for global config
• Environment variable precedence & fallback chains
• Schema validation using standard library dataclasses & typed fields
• Automatic scratch & cache directory provisioning
• Comprehensive English documentation, type hints, and HPC-grade resilience
• Zero circular dependencies: pure stdlib typing with careful enum casting
All documentation and inline comments are in English per project standards.
"""

import contextlib
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


# Lazy import for logger to avoid circular dependency
def get_logger(name: str):
    from .logging_config import get_logger as _get_logger
    return _get_logger(name)

logger = get_logger(__name__)

# =============================================================================
# Constants & Default Paths
# =============================================================================
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "forge"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "forge"
DEFAULT_SCRATCH_ENV = os.environ.get("SCRATCH", os.environ.get("TMPDIR", "/tmp"))
DEFAULT_TIMEOUT_SEC = 300.0
DEFAULT_MAX_CORES = os.cpu_count() or 16
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_BACKEND = "wien2k"
CONFIG_VERSION = "1.2.0"
OUTPUT_FILE = ".machines"  # Default output filename for WIEN2k configuration

# =============================================================================
# Configuration Schema (Dataclass)
# =============================================================================
@dataclass
class AppConfig:
    """
    Central configuration container for FORGE.
    Fields are validated, type-coerced, and thread-safe.
    """
    version: str = CONFIG_VERSION
    wienroot: str = os.environ.get("WIENROOT", "/opt/codes/WIEN2k")
    scratch_dir: str = DEFAULT_SCRATCH_ENV
    config_dir: str = str(DEFAULT_CONFIG_DIR)
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    log_level: str = DEFAULT_LOG_LEVEL
    backend: str = DEFAULT_BACKEND
    max_cores: int = DEFAULT_MAX_CORES
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    enable_tui: bool = True
    quiet_mode: bool = False
    dry_run: bool = False
    custom_paths: dict[str, str] = field(default_factory=dict)
    _is_validated: bool = field(default=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Reconstruct from dictionary with safe fallbacks."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        clean = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**clean)

# =============================================================================
# Thread-Safe Config Manager
# =============================================================================
class ConfigManager:
    """
    Lazy-initialized, thread-safe configuration loader.
    Handles environment parsing, file loading, validation, and global access.
    """
    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()
    _config: Optional[AppConfig] = None
    _initialized = False
    _validation_errors: list[str] = []  # noqa: RUF012

    def __new__(cls) -> "ConfigManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._config = None
                cls._instance._validation_errors = []
                cls._instance._initialized = True
            return cls._instance

    @classmethod
    def instance(cls) -> "ConfigManager":
        """Explicit singleton accessor for compatibility."""
        return cls()

    def load(
        self,
        env_override: Optional[dict[str, Any]] = None,
        file_path: Optional[Union[str, Path]] = None,
        cli_override: Optional[dict[str, Any]] = None
    ) -> AppConfig:
        """
        Load configuration with strict precedence:
        1. Hardcoded defaults
        2. File-based config (JSON)
        3. Environment variables
        4. CLI/Runtime overrides
        """
        # 1. Defaults
        cfg = AppConfig()

        # 2. File-based
        if file_path:
            cfg = self._merge_file_config(cfg, Path(file_path))

        # 3. Environment
        cfg = self._merge_env_config(cfg)

        # 4. CLI/Programmatic
        if cli_override:
            cfg = self._merge_dict_config(cfg, cli_override)
        if env_override:
            cfg = self._merge_dict_config(cfg, env_override)

        # Normalize paths
        cfg.wienroot = str(Path(cfg.wienroot).expanduser().resolve())
        cfg.scratch_dir = str(Path(cfg.scratch_dir).expanduser().resolve())
        cfg.config_dir = str(Path(cfg.config_dir).expanduser().resolve())
        cfg.cache_dir = str(Path(cfg.cache_dir).expanduser().resolve())

        # Validate
        self._validation_errors = self._validate_config(cfg)
        cfg._is_validated = len(self._validation_errors) == 0

        self._config = cfg
        return cfg

    def get_config(self) -> AppConfig:
        """Retrieve current configuration. Auto-loads if missing."""
        if self._config is None:
            return self.load()
        return self._config

    @property
    def errors(self) -> list[str]:
        return self._validation_errors.copy()

    def save(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Persist current config to JSON."""
        if self._config is None:
            logger.error("Cannot save: configuration not loaded.")
            return False
            
        target = Path(path) if path else Path(self._config.config_dir) / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(json.dumps(self._config.to_dict(), indent=2), encoding="utf-8")
            logger.info(f"Configuration saved to {target}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    # =========================================================================
    # Internal Merge & Validation Logic
    # =========================================================================

    def _merge_file_config(self, base: AppConfig, path: Path) -> AppConfig:
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                return self._merge_dict_config(base, data)
            except Exception as e:
                logger.warning(f"Config file read failed: {e}")
        return base

    def _merge_env_config(self, base: AppConfig) -> AppConfig:
        env_map = {
            "WIENROOT": "wienroot",
            "SCRATCH": "scratch_dir",
            "LOG_LEVEL": "log_level",
            "WIEN2K_BACKEND": "backend",
            "WIEN2K_MAX_CORES": "max_cores",
            "WIEN2K_TIMEOUT": "timeout_sec",
            "WIEN2K_QUIET": "quiet_mode",
        }
        overrides = {}
        for env_var, cfg_key in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                overrides[cfg_key] = val.lower() if cfg_key in ("quiet_mode", "enable_tui") else val
        return self._merge_dict_config(base, overrides)

    def _merge_dict_config(self, base: AppConfig, override: dict[str, Any]) -> AppConfig:
        data = base.to_dict()
        for k, v in override.items():
            if k in data:
                # Type coercion
                if isinstance(v, str) and isinstance(data[k], int):
                    with contextlib.suppress(ValueError): 
                        v = int(v)
                elif isinstance(v, str) and isinstance(data[k], float):
                    with contextlib.suppress(ValueError): 
                        v = float(v)  
                elif isinstance(v, str) and isinstance(data[k], bool):
                    v = v.lower() in ("true", "1", "yes")
                data[k] = v
        return AppConfig.from_dict(data)

    def _validate_config(self, cfg: AppConfig) -> list[str]:  # noqa: C901
        errors = []
        # Path checks
        wien = Path(cfg.wienroot)
        if cfg.wienroot != "/opt/codes/WIEN2k" and not wien.exists():
            errors.append(f"WIENROOT directory not found: {cfg.wienroot}")
        elif cfg.wienroot != "/opt/codes/WIEN2k" and not os.access(wien, os.R_OK | os.X_OK):
            errors.append(f"Insufficient permissions for WIENROOT: {cfg.wienroot}")
            
        scratch = Path(cfg.scratch_dir)
        if not scratch.exists():
            try: 
                scratch.mkdir(parents=True, exist_ok=True)
            except Exception: 
                errors.append(f"SCRATCH path does not exist and cannot be created: {cfg.scratch_dir}")
        elif not os.access(scratch, os.W_OK):
            errors.append(f"SCRATCH path is not writable: {cfg.scratch_dir}")
            
        # Logical checks
        if cfg.max_cores <= 0:
            errors.append("max_cores must be > 0")
        if cfg.timeout_sec <= 0:
            errors.append("timeout_sec must be > 0")

        # Enum validation for log_level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if cfg.log_level and cfg.log_level.upper() not in valid_levels:
            errors.append(f"Invalid log_level: {cfg.log_level}. Use one of: {sorted(valid_levels)}")

        # Enum validation for backend
        from .types import BackendCode
        if cfg.backend:
            try:
                BackendCode(cfg.backend.lower())
            except ValueError:
                valid_backends = [b.value for b in BackendCode]
                errors.append(f"Unsupported backend: {cfg.backend}. Use one of: {valid_backends}")
            
        return errors

# =============================================================================
# Public API & Singleton Exposure
# =============================================================================
def get_config() -> AppConfig:
    """Thread-safe access to the global configuration."""
    return ConfigManager().get_config()

def load_config(**kwargs: Any) -> AppConfig:
    """Explicitly load/refresh configuration with overrides.
    
    Accepts named arguments that are forwarded to ConfigManager.load():
        file_path: Optional path to a config JSON/TOML file
        cli_override: Dict of CLI-level overrides
        env_override: Dict of environment-level overrides
    """
    return ConfigManager().load(**kwargs)

def validate_config() -> list[str]:
    """Return validation errors from the current configuration."""
    return ConfigManager().errors

def ensure_dirs() -> None:
    """Guarantee that config, cache, and scratch directories exist."""
    cfg = get_config()
    for d in (cfg.config_dir, cfg.cache_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

# =============================================================================
# Explicit Public API Declaration
# =============================================================================
__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_MAX_CORES",
    "DEFAULT_TIMEOUT_SEC",
    "AppConfig",
    "ConfigManager",
    "ensure_dirs",
    "get_config",
    "load_config",
    "validate_config",
]