"""
Backward-compatible re-export shim. Delegates to forge.backends (unified registry).

All callers should migrate to `from forge.backends import ...` over time.
This module exists solely for backward compatibility with existing import sites.
"""

from typing import Optional, Union

from .backends import (  # noqa: F401 — re-export for callers
    Backend,
    auto_detect,
    get_backend,
    get_backend_class,
    get_current_backend,
    is_backend_available,
    list_backends,
    reset,
    set_backend,
)
from .types import BackendCode


class BackendManager:
    """Backward-compatible singleton wrapper. Delegates to forge.backends directly."""

    _instance: Optional["BackendManager"] = None

    def __new__(cls) -> "BackendManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def instance(cls) -> "BackendManager":
        return cls()

    @staticmethod
    def auto_detect() -> BackendCode:
        return auto_detect()

    @staticmethod
    def get_backend_class(code: Optional[Union[BackendCode, str]] = None) -> type:
        return get_backend_class(code)

    @staticmethod
    def get_backend(code: Optional[Union[BackendCode, str]] = None) -> Backend:
        return get_backend(code)

    @staticmethod
    def set_backend(code: Union[BackendCode, str]) -> None:
        set_backend(code)

    @staticmethod
    def list_available() -> list[BackendCode]:
        return list_backends()

    @staticmethod
    def reset() -> None:
        reset()


__all__ = [
    "Backend",
    "BackendCode",
    "BackendManager",
    "auto_detect",
    "get_backend",
    "get_backend_class",
    "get_current_backend",
    "list_backends",
    "set_backend",
]
