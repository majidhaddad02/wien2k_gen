"""
Unified backend registry, factory, and instance manager for DFT codes.
Merged from backends/__init__.py + backend_manager.py — single source of truth.

Key Architecture Features:
• Thread-safe singleton registry with RLock for concurrent CLI/TUI access
• Lazy module loading with importlib to minimize startup overhead & circular imports
• Auto-detection based on input file signatures in the working directory
• Stub backend pattern for optional dependencies with actionable error messages
• Instance caching with thread-safe recreation on backend switch
• Strict type safety using BackendCode enum + Backend ABC contract
• PEP 562 lazy attribute access for concrete backend classes
• Full backward compatibility with forge.backend_manager import sites

All documentation and inline comments are in English per project standards.
"""

import importlib
import threading
from pathlib import Path
from typing import Any, Optional, Union

from ..exceptions import BackendError, MissingInputError
from ..logging_config import get_logger
from ..types import BackendCode
from .base import Backend

logger = get_logger(__name__)

# =============================================================================
# Thread-Safe Global State
# =============================================================================

_BACKENDS: dict[BackendCode, type[Backend]] = {}
_REGISTRY_LOCK = threading.RLock()
_LOADED = False
_current_code: Optional[BackendCode] = None
_cached_instance: Optional[Backend] = None


def _key_to_code(key: str) -> Optional[BackendCode]:
    """Map normalized string key to BackendCode enum."""
    key = key.lower().strip().replace("-", " ").replace("  ", "_")
    mapping = {
        "wien2k": BackendCode.WIEN2K,
        "qe": BackendCode.QUANTUM_ESPRESSO,
        "quantum_espresso": BackendCode.QUANTUM_ESPRESSO,
        "vasp": BackendCode.VASP,
        "cp2k": BackendCode.CP2K,
    }
    return mapping.get(key)


def _str_or_code(code: Union[BackendCode, str, None]) -> Optional[BackendCode]:
    """Coerce string/None to BackendCode enum."""
    if isinstance(code, BackendCode):
        return code
    if isinstance(code, str):
        return _key_to_code(code)
    return None


# =============================================================================
# Stub Backend Factory
# =============================================================================

def _make_stub_backend(code: BackendCode, class_name: str, error: Optional[Exception] = None) -> type[Backend]:
    """Create stub that raises clear, actionable error on instantiation."""
    stub_module = f"forge.backends.{code.value}"

    class StubBackend(Backend):
        _is_stub: bool = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            msg = (
                f"{code.value} backend is unavailable. "
                f"Install optional dependencies: pip install wien2k-gen[{code.value}]"
            )
            if error:
                msg += f"\nUnderlying error: {error}"
            raise BackendError(msg) from error

        def detect_problem_size(self) -> dict[str, Any]:
            raise BackendError(f"{code.value} backend is unavailable. Install optional dependencies.")

        def generate_input(self, topo: Any, suggestion: Any) -> str:
            raise BackendError(f"{code.value} backend is unavailable. Install optional dependencies.")

        def get_execution_command(self, suggestion: Any) -> str:
            raise BackendError(f"{code.value} backend is unavailable. Install optional dependencies.")

    StubBackend.__name__ = class_name
    StubBackend.__module__ = stub_module
    return StubBackend


# =============================================================================
# Core Registry — Lazy Loading with Double-Checked Locking
# =============================================================================

def _load_backends() -> None:
    """Lazy-load all backend classes with graceful dependency fallback."""
    global _LOADED, _BACKENDS

    if _LOADED:
        return

    with _REGISTRY_LOCK:
        if _LOADED:
            return

        # WIEN2k is the mandatory primary backend
        try:
            from .wien2k import Wien2kBackend
            _BACKENDS[BackendCode.WIEN2K] = Wien2kBackend
        except ImportError as e:
            raise ImportError(
                "Critical backend 'wien2k' could not be imported. "
                "Verify WIENROOT, installation, and all required dependencies."
            ) from e

        # Optional backends with graceful fallback to stubs
        optional_map: dict[BackendCode, tuple[str, str]] = {
            BackendCode.VASP: (".vasp", "VaspBackend"),
            BackendCode.QUANTUM_ESPRESSO: (".quantum_espresso.backend", "QuantumEspressoBackend"),
            BackendCode.CP2K: (".cp2k", "CP2KBackend"),
        }

        for code, (module_path, class_name) in optional_map.items():
            try:
                module = importlib.import_module(module_path, package=__package__)
                cls_obj = getattr(module, class_name)
                _BACKENDS[code] = cls_obj
            except ImportError as e:
                logger.warning(f"Backend '{code.value}' skipped due to missing dependencies: {e}")
                _BACKENDS[code] = _make_stub_backend(code, class_name, e)
            except Exception as e:
                logger.error(f"Failed to load backend '{code.value}': {e}")
                _BACKENDS[code] = _make_stub_backend(code, class_name, e)

        _LOADED = True


# =============================================================================
# Auto-Detection
# =============================================================================

def auto_detect() -> BackendCode:
    """Detect backend from working directory input file signatures."""
    cwd = Path.cwd()
    signatures: dict[BackendCode, list] = {
        BackendCode.WIEN2K: [cwd.glob("*.struct")],
        BackendCode.VASP: [cwd.glob("POSCAR*"), cwd.glob("INCAR*")],
        BackendCode.QUANTUM_ESPRESSO: [cwd.glob("*.pw.in"), cwd.glob("*.in")],
        BackendCode.CP2K: [cwd.glob("*.inp")],
    }

    matches: list[BackendCode] = []
    for code, globs in signatures.items():
        for g in globs:
            if any(True for _ in g):
                matches.append(code)
                break

    if not matches:
        raise MissingInputError(
            "No recognizable DFT input files found in current directory. "
            "Specify backend explicitly via CLI/TUI or run in a project directory."
        )

    priority = [BackendCode.WIEN2K, BackendCode.VASP, BackendCode.QUANTUM_ESPRESSO, BackendCode.CP2K]
    detected = next((p for p in priority if p in matches), matches[0])
    logger.info(f"Auto-detected backend from filesystem: {detected.value}")
    return detected


# =============================================================================
# Public API — Class Access
# =============================================================================

def get_backend_class(code: Optional[Union[BackendCode, str]] = None) -> type[Backend]:
    """Return backend class by code. Auto-loads registry on first call."""
    _load_backends()

    target_code = _str_or_code(code)
    if target_code is None:
        target_code = auto_detect()

    if target_code not in _BACKENDS:
        raise BackendError(f"Unsupported or unavailable backend: {target_code.value}")

    cls_obj = _BACKENDS[target_code]
    if getattr(cls_obj, "_is_stub", False):
        try:
            cls_obj()
        except Exception as e:
            raise BackendError(f"Backend '{target_code.value}' is unavailable: {e}") from e
    return cls_obj


# =============================================================================
# Public API — Instance Access (Cached, Thread-Safe)
# =============================================================================

def get_backend(code: Optional[Union[BackendCode, str]] = None) -> Backend:
    """Return cached backend instance. Creates new one on first call or after switch."""
    global _current_code, _cached_instance

    target_code = _str_or_code(code)
    if target_code is None:
        target_code = auto_detect()

    with _REGISTRY_LOCK:
        if _current_code == target_code and _cached_instance is not None:
            return _cached_instance

        cls_obj = get_backend_class(target_code)
        instance = cls_obj()
        _cached_instance = instance
        _current_code = target_code
        logger.info(f"Backend instance initialized: {target_code.value}")
        return instance


def get_current_backend() -> Backend:
    """Return the currently active backend instance."""
    global _current_code, _cached_instance

    _load_backends()

    with _REGISTRY_LOCK:
        if _cached_instance is not None and _current_code is not None:
            return _cached_instance

        target_code = _guess_active_code()
        cls_obj = get_backend_class(target_code)
        instance = cls_obj()
        _cached_instance = instance
        _current_code = target_code
        return instance


def _guess_active_code() -> BackendCode:
    """Pick the active backend code (auto-detect or first available)."""
    try:
        return auto_detect()
    except MissingInputError:
        available = list_backends()
        if available:
            return available[0]
        return BackendCode.WIEN2K


def set_backend(code: Union[BackendCode, str]) -> None:
    """Explicitly switch active backend and invalidate cached instance."""
    global _current_code, _cached_instance

    target_code = _str_or_code(code)
    with _REGISTRY_LOCK:
        get_backend_class(target_code)
        _current_code = target_code
        _cached_instance = None
        logger.info(f"Backend switched to: {target_code.value}")


def list_backends() -> list[BackendCode]:
    """Return list of successfully loaded (non-stub) backend codes."""
    _load_backends()
    with _REGISTRY_LOCK:
        return [code for code, cls in _BACKENDS.items() if not getattr(cls, "_is_stub", False)]


def is_backend_available(code: Union[BackendCode, str]) -> bool:
    """Check if a backend is available and fully functional (not a dependency stub)."""
    target_code = _str_or_code(code)
    if target_code is None:
        return False
    _load_backends()
    with _REGISTRY_LOCK:
        backend_class = _BACKENDS.get(target_code)
        return backend_class is not None and not getattr(backend_class, "_is_stub", False)


def reset() -> None:
    """Clear all caches and reload registry (for testing, signal handlers, reload)."""
    global _LOADED, _BACKENDS, _current_code, _cached_instance
    with _REGISTRY_LOCK:
        _LOADED = False
        _BACKENDS = {}
        _current_code = None
        _cached_instance = None


# =============================================================================
# Public API Declaration
# =============================================================================

__all__ = [
    "Backend",
    "auto_detect",
    "get_backend",
    "get_backend_class",
    "get_current_backend",
    "is_backend_available",
    "list_backends",
    "reset",
    "set_backend",
]


# =============================================================================
# PEP 562 — Lazy Attribute Access for Concrete Backend Classes
# =============================================================================

def __getattr__(name: str):
    """Lazy attr: `from forge.backends import Wien2kBackend` without upfront import."""
    _load_backends()
    mapping = {
        "Wien2kBackend": BackendCode.WIEN2K,
        "QuantumEspressoBackend": BackendCode.QUANTUM_ESPRESSO,
        "QEBackend": BackendCode.QUANTUM_ESPRESSO,
        "VaspBackend": BackendCode.VASP,
        "CP2KBackend": BackendCode.CP2K,
    }
    if name in mapping:
        return _BACKENDS.get(mapping[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
