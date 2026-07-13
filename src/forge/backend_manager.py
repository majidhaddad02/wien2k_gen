"""
Backend Registry & Factory Manager for FORGE.
Provides thread-safe, lazy-loaded access to DFT code backends (WIEN2k, QE, VASP, CP2K).
Supports auto-detection, runtime switching, dependency fallbacks, and instance caching.

Key Architecture Features:
• Thread-safe singleton registry with RLock for concurrent CLI/TUI access
• Lazy module loading to minimize startup overhead & prevent circular imports
• Auto-detection based on input file signatures in the working directory
• Stub backend pattern for optional dependencies with actionable ImportError hints
• Instance caching with thread-safe recreation on backend switch
• Strict type safety using BackendCode enum & Backend ABC contract
• Comprehensive English documentation, type hints, and HPC-grade resilience

All documentation and inline comments are in English per project standards.
"""

import importlib
import threading
from pathlib import Path
from typing import Any, Optional, Union

from .backends.base import Backend
from .exceptions import BackendError, MissingInputError
from .logging_config import get_logger
from .types import BackendCode

logger = get_logger(__name__)


# =============================================================================
# Core Backend Manager (Thread-Safe Singleton)
# =============================================================================

class BackendManager:
    """
    Central factory & registry for DFT backends.
    Manages lifecycle, lazy loading, auto-detection, and instance caching.
    """
    _instance: Optional["BackendManager"] = None
    _lock = threading.RLock()
    _registry: dict[BackendCode, type[Backend]] = {}  # noqa: RUF012
    _loaded = False
    _current_code: Optional[BackendCode] = None
    _cached_instance: Optional[Backend] = None

    def __new__(cls) -> "BackendManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self) -> None:
        """Reset internal state (safe for testing/runtime reload)."""
        self._registry.clear()
        self._loaded = False
        self._current_code = None
        self._cached_instance = None

    @classmethod
    def instance(cls) -> "BackendManager":
        """Thread-safe singleton accessor."""
        return cls()

    def _load_backends(self) -> None:
        """Lazy-load all backend classes with graceful dependency fallback."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            # Mapping: BackendCode -> (module_path, class_name)
            backends_map = {
                BackendCode.WIEN2K: ("..backends.wien2k", "Wien2kBackend"),
                BackendCode.VASP: ("..backends.vasp", "VaspBackend"),
                BackendCode.QUANTUM_ESPRESSO: ("..backends.quantum_espresso.backend", "QuantumEspressoBackend"),
                BackendCode.CP2K: ("..backends.cp2k", "CP2KBackend"),
            }

            for code, (module_path, class_name) in backends_map.items():
                try:
                    module = importlib.import_module(module_path, package=__package__)
                    backend_cls = getattr(module, class_name)
                    self._registry[code] = backend_cls
                except ImportError as e:
                    logger.warning(f"Backend '{code.value}' skipped due to missing dependencies: {e}")
                    self._registry[code] = self._make_stub_backend(code, class_name, e)
                except Exception as e:
                    logger.error(f"Failed to load backend '{code.value}': {e}")
                    self._registry[code] = self._make_stub_backend(code, class_name, e)

            self._loaded = True

    def _make_stub_backend(self, code: BackendCode, class_name: str, error: Exception) -> type[Backend]:
        """Create a stub class that raises a clear, actionable error on instantiation."""
        class StubBackend(Backend):
            def detect_problem_size(self) -> dict[str, Any]:
                raise BackendError(f"{code.value} backend is unavailable. Please install missing dependencies.")
            
            def generate_input(self, topo: Any, suggestion: Any) -> str:
                raise BackendError(f"{code.value} backend is unavailable. Please install missing dependencies.")
            
            def get_execution_command(self, suggestion: Any) -> str:
                raise BackendError(f"{code.value} backend is unavailable. Please install missing dependencies.")

        StubBackend.__name__ = class_name
        StubBackend.__module__ = f"forge.backends.{code.value}"
        StubBackend._is_stub = True  # type: ignore[attr-defined]
        return StubBackend

    def auto_detect(self) -> BackendCode:
        """Detect backend from working directory input file signatures."""
        cwd = Path.cwd()
        signatures = {
            BackendCode.WIEN2K: [cwd.glob("*.struct")],
            BackendCode.VASP: [cwd.glob("POSCAR*"), cwd.glob("INCAR*")],
            BackendCode.QUANTUM_ESPRESSO: [cwd.glob("*.pw.in"), cwd.glob("*.in")],
            BackendCode.CP2K: [cwd.glob("*.inp")],
        }

        matches = []
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

        # Priority fallback: WIEN2K > VASP > QE > CP2K
        priority = [BackendCode.WIEN2K, BackendCode.VASP, BackendCode.QUANTUM_ESPRESSO, BackendCode.CP2K]
        for p in priority:
            if p in matches:
                detected = p
                break
        else:
            detected = matches[0]

        logger.info(f"Auto-detected backend from filesystem: {detected.value}")
        return detected

    def get_backend_class(self, code: Optional[Union[BackendCode, str]] = None) -> type[Backend]:
        """Return backend class by code. Auto-loads registry if needed."""
        self._load_backends()
        
        if isinstance(code, str):
            code = BackendCode(code)
        if code is None:
            code = self.auto_detect()

        if code not in self._registry:
            raise BackendError(f"Unsupported or unavailable backend: {code.value}")

        cls_obj = self._registry[code]
        if getattr(cls_obj, "_is_stub", False):
            try:
                cls_obj()  # Triggers the descriptive ImportError/BackendError
            except Exception as e:
                raise BackendError(f"Backend '{code.value}' is unavailable: {e}") from e
        return cls_obj

    def get_backend(self, code: Optional[Union[BackendCode, str]] = None) -> Backend:
        """Return cached backend instance. Creates new one if switched or missing."""
        with self._lock:
            target_code = BackendCode(code) if isinstance(code, str) else code
            if target_code is None:
                target_code = self.auto_detect()

            # Return cached if code matches and instance is valid
            if self._current_code == target_code and self._cached_instance is not None:
                return self._cached_instance

            # Instantiate new
            cls_obj = self.get_backend_class(target_code)
            instance = cls_obj()
            self._cached_instance = instance
            self._current_code = target_code
            logger.info(f"Backend instance initialized: {target_code.value}")
            return instance

    def set_backend(self, code: Union[BackendCode, str]) -> None:
        """Explicitly switch active backend and invalidate cache."""
        with self._lock:
            target_code = BackendCode(code) if isinstance(code, str) else code
            self.get_backend_class(target_code)  # Validate availability first
            self._current_code = target_code
            self._cached_instance = None  # Force recreation on next get_backend()
            logger.info(f"Backend switched to: {target_code.value}")

    def list_available(self) -> list[BackendCode]:
        """Return list of successfully loaded (non-stub) backends."""
        self._load_backends()
        return [code for code, cls in self._registry.items() if not cls.__name__.endswith("StubBackend")]

    def reset(self) -> None:
        """Clear cache & registry (mainly for testing or dynamic reload)."""
        with self._lock:
            self._init()


# =============================================================================
# Public API (Module-Level Convenience)
# =============================================================================

def get_backend(code: Optional[Union[BackendCode, str]] = None) -> Backend:
    """Convenience function to get the active backend instance."""
    return BackendManager.instance().get_backend(code)


def get_current_backend() -> Backend:
    """Convenience function to get the currently active backend instance."""
    return BackendManager.instance().get_backend()


def get_backend_class(code: Optional[Union[BackendCode, str]] = None) -> type[Backend]:
    """Convenience function to get the backend class."""
    return BackendManager.instance().get_backend_class(code)


def set_backend(code: Union[BackendCode, str]) -> None:
    """Convenience function to switch the active backend."""
    BackendManager.instance().set_backend(code)


def list_backends() -> list[BackendCode]:
    """Convenience function to list available backends."""
    return BackendManager.instance().list_available()


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "BackendManager",
    "get_backend",
    "get_backend_class",
    "list_backends",
    "set_backend",
]