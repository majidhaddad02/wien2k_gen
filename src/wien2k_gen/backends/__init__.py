"""
Backend registry and factory for DFT codes.
Production features:
• Lazy loading to minimize import overhead and prevent dependency conflicts at startup
• Factory pattern with auto-detection based on input file signatures in CWD
• Thread-safe registration with RLock for concurrent CLI/UI access
• Graceful fallback for optional backends via stub classes with actionable ImportError
• Consistent naming conventions and runtime backend switching support
• PEP 562 compatible lazy attribute resolution via __getattr__
• Comprehensive type hints and explicit __all__ API boundary
All documentation and inline comments are in English per project standards.
"""

import importlib
import threading
from pathlib import Path
from typing import Optional

from ..logging_config import get_logger
from .base import Backend

logger = get_logger(__name__)

# =============================================================================
# Thread-Safe Global Registry
# =============================================================================
_BACKENDS: dict[str, type[Backend]] = {}
_REGISTRY_LOCK = threading.RLock()
_LOADED = False


def _normalize_name(name: str) -> str:
    """Normalize backend name for consistent lookup (case-insensitive, hyphen-safe)."""
    return name.lower().strip().replace("-", " ").replace("  ", "_")


def _make_stub_backend(key: str, class_name: str) -> type[Backend]:
    """
    Create a stub backend class that raises a helpful ImportError on instantiation.
    Used for optional backends where external dependencies are missing.
    """
    class StubBackend(Backend):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{class_name} backend requires additional dependencies. "
                f"Please install optional dependencies: pip install wien2k-gen[{key}]"
            )

        # Satisfy ABC requirements with explicit NotImplementedError
        def detect_problem_size(self) -> dict:
            raise NotImplementedError("Stub backend cannot detect problem size.")

        def generate_input(self, topo, suggestion) -> str:
            raise NotImplementedError("Stub backend cannot generate input.")

        def get_execution_command(self, suggestion) -> str:
            raise NotImplementedError("Stub backend cannot provide execution command.")

    StubBackend.__name__ = class_name
    StubBackend.__module__ = f"{__package__}.{key}"
    return StubBackend


def _load_backends() -> None:
    """
    Lazy-load backend classes with error handling for optional dependencies.
    Thread-safe with double-checked locking pattern.
    """
    global _LOADED, _BACKENDS
    with _REGISTRY_LOCK:
        if _LOADED:
            return
            
        # Register core backend (WIEN2k is mandatory for this package)
        try:
            from .wien2k import Wien2kBackend
            _BACKENDS["wien2k"] = Wien2kBackend
        except ImportError as e:
            raise ImportError(
                "Critical backend 'wien2k' could not be imported. "
                f"Original error: {e}. Please verify WIENROOT and installation."
            ) from e

        # Register optional backends with graceful fallback
        optional_backends = [
            ("quantum_espresso", "QuantumEspressoBackend", "quantum_espresso"),
            ("vasp", "VaspBackend", "vasp"),
            ("cp2k", "CP2KBackend", "cp2k"),
        ]

        for module_path, class_name, key in optional_backends:
            try:
                module = importlib.import_module(f".{module_path}", package=__package__)
                backend_class = getattr(module, class_name)
                _BACKENDS[key] = backend_class
                # Add convenient aliases
                if key == "quantum_espresso":
                    _BACKENDS["qe"] = backend_class
            except ImportError:
                # Dependency missing; register stub that raises helpful error on use
                _BACKENDS[key] = _make_stub_backend(key, class_name)

        _LOADED = True


def get_backend(code: Optional[str] = None) -> type[Backend]:
    """
    Factory function to retrieve backend class by code name.
    Supports auto-detection based on available input files in the current directory.
    """
    _load_backends()

    if code is None:
        # Auto-detect based on input file signatures in CWD
        if list(Path(".").glob("*.struct")):
            code = "wien2k"
        elif list(Path(".").glob("*.in")) and list(Path(".").glob("*.pw.in*")):
            code = "quantum_espresso"
        elif list(Path(".").glob("INCAR")) and list(Path(".").glob("POSCAR")):
            code = "vasp"
        elif list(Path(".").glob("*.inp")):
            code = "cp2k"
        else:
            available = ", ".join(k for k in _BACKENDS if not k.startswith("_"))
            raise ValueError(
                f"Could not auto-detect DFT code. Please specify code= one of: {available}. "
                f"Or run in a directory with appropriate input files."
            )

    code = _normalize_name(code)

    with _REGISTRY_LOCK:
        if code not in _BACKENDS:
            available = ", ".join(_BACKENDS.keys())
            raise ValueError(f"Unknown backend code: '{code}'. Available: {available}")

        backend_class = _BACKENDS[code]

        # Eagerly fail if it's a stub to provide immediate, actionable feedback
        if backend_class.__name__ == "StubBackend":
            try:
                backend_class()  # Triggers ImportError with dependency instructions
            except ImportError as e:
                raise e from None

        return backend_class


def list_backends() -> list[str]:
    """Return sorted list of fully available backend names (excluding stubs)."""
    _load_backends()
    with _REGISTRY_LOCK:
        return sorted([
            k for k, v in _BACKENDS.items()
            if v.__name__ != "StubBackend" and not k.startswith("_")
        ])


def is_backend_available(code: str) -> bool:
    """Check if a backend is available and fully functional (not a dependency stub)."""
    _load_backends()
    code = _normalize_name(code)
    with _REGISTRY_LOCK:
        backend_class = _BACKENDS.get(code)
        return backend_class is not None and backend_class.__name__ != "StubBackend"


# =============================================================================
# Explicit Public API Declaration
# =============================================================================
__all__ = [
    "Backend",
    "CP2KBackend",
    "QuantumEspressoBackend",
    "VaspBackend",
    "Wien2kBackend",
    "get_backend",
    "is_backend_available",
    "list_backends",
]


# =============================================================================
# Lazy Attribute Access (PEP 562)
# =============================================================================
def __getattr__(name: str):
    """
    Lazy attribute access for concrete backend classes.
    Enables `from wien2k_gen.backends import Wien2kBackend` without upfront import cost.
    """
    _load_backends()
    mapping = {
        "Wien2kBackend": "wien2k",
        "QuantumEspressoBackend": "quantum_espresso",
        "QEBackend": "quantum_espresso",
        "VaspBackend": "vasp",
        "CP2KBackend": "cp2k",
    }
    if name in mapping:
        return _BACKENDS.get(mapping[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")