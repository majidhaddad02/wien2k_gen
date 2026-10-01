"""
Centralized external dependency locator for FORGE.
Auto-detects WIEN2k, ELPA, CP2K installation paths at runtime
using PATH scanning and environment variables -- no hardcoded paths.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

_ELPA_ENV_VARS = ("ELPA_DIR", "ELPA_HOME", "ELPA_ROOT")
_ELPA_LIB_GLOBS = ("libelpa*.so*", "libelpa*.a")
_PKG_CONFIG_TIMEOUT = 5


def find_wienroot() -> Optional[str]:
    """Detect WIEN2k root directory.

    Resolution order:
      1. ``WIENROOT`` environment variable
      2. Scan PATH for ``run_lapw`` binary and resolve to parent
      3. Scan PATH for ``siteconfig_lapw`` binary
      4. Return None if not found

    Returns:
        Absolute path to WIEN2k installation root, or None.
    """
    env = os.environ.get("WIENROOT")
    if env and Path(env).is_dir():
        return str(Path(env).resolve())

    for binary in ("run_lapw", "siteconfig_lapw"):
        exe = shutil.which(binary)
        if exe:
            root = Path(exe).resolve().parent
            if root.is_dir():
                return str(root)

    return None


def _contains_elpa_lib(directory: Path) -> bool:
    """Return True if *directory* itself holds a ``libelpa*`` shared/static library."""
    if not directory.is_dir():
        return False
    return any(next(directory.glob(pattern), None) is not None for pattern in _ELPA_LIB_GLOBS)


def _verified_elpa_prefix(candidate: Optional[str]) -> Optional[str]:
    """Accept *candidate* only if it plausibly contains an ELPA library.

    Looks in ``lib/`` and ``lib64/`` under the prefix, and also treats the
    candidate itself as a library directory (``.../lib`` or ``.../lib64``).
    """
    if not candidate:
        return None
    prefix = Path(candidate)
    if not prefix.is_dir():
        return None
    for libdir_name in ("lib", "lib64"):
        if _contains_elpa_lib(prefix / libdir_name):
            return str(prefix.resolve())
    if _contains_elpa_lib(prefix):
        if prefix.name in ("lib", "lib64"):
            return str(prefix.parent.resolve())
        return str(prefix.resolve())
    return None


def _from_pkg_config() -> Optional[str]:
    """Locate ELPA via pkg-config (.pc files: elpa, elpa_openmp, elpa-<ver>)."""
    try:
        listed = subprocess.run(
            ["pkg-config", "--list-all"],
            capture_output=True,
            text=True,
            timeout=_PKG_CONFIG_TIMEOUT,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if listed.returncode != 0 or not listed.stdout:
        return None

    matched_name = None
    for line in listed.stdout.splitlines():
        name = line.split(None, 1)[0] if line.strip() else ""
        if name.lower().startswith("elpa"):
            matched_name = name
            break
    if not matched_name:
        return None

    try:
        prefix_result = subprocess.run(
            ["pkg-config", "--variable=prefix", matched_name],
            capture_output=True,
            text=True,
            timeout=_PKG_CONFIG_TIMEOUT,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return _verified_elpa_prefix(prefix_result.stdout.strip())


def _from_linker_paths() -> Optional[str]:
    """Scan ``LD_LIBRARY_PATH`` then ``LIBRARY_PATH`` for ``libelpa*``."""
    for env_name in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        raw = os.environ.get(env_name, "")
        for entry in raw.split(":"):
            if not entry:
                continue
            libdir = Path(entry)
            if not _contains_elpa_lib(libdir):
                continue
            if libdir.name in ("lib", "lib64"):
                return str(libdir.parent.resolve())
            return str(libdir.resolve())
    return None


def find_elpa_dir() -> Optional[str]:
    """Detect ELPA library installation directory.

    Layered resolution, strongest signal first. Each candidate is verified
    by a shallow ``lib*/libelpa*`` file check before being accepted — a
    directory that merely exists is not enough. This is deliberately
    portable across HPC sites (manual installs, EasyBuild, Spack, distro
    packages, ``module load``) rather than tied to one cluster layout.

    Resolution order:
      1. Explicit user env vars ``ELPA_DIR``, ``ELPA_HOME``, ``ELPA_ROOT``
         (several common spellings; ``ELPA_DIR`` is what compile-flag
         helpers historically read).
      2. EasyBuild ``EBROOTELPA`` — set automatically when an ELPA module
         built with EasyBuild is loaded (common at European HPC centres).
      3. pkg-config: ``pkg-config --list-all`` for any package name
         starting with ``elpa``, then ``--variable=prefix``. Tool-agnostic;
         skipped cleanly if pkg-config is absent.
      4. Linker search paths: scan each entry of ``LD_LIBRARY_PATH`` then
         ``LIBRARY_PATH`` for ``libelpa*.so*`` / ``libelpa*.a`` and derive
         the install prefix as the parent of ``lib`` / ``lib64``. Typical
         after ``module load elpa`` when no ``*_HOME`` variable is exported.
      5. ``None`` if nothing above yields a verified installation.

    A MODULEPATH scan is intentionally omitted: ``$MODULEPATH`` holds
    Lmod/Tcl modulefiles, not library prefixes, so looking for an ``ELPA``
    subdirectory there is a known-unreliable heuristic.

    Returns:
        Absolute path to ELPA installation prefix, or None.
    """
    for var in _ELPA_ENV_VARS:
        found = _verified_elpa_prefix(os.environ.get(var))
        if found:
            logger.debug("ELPA found via %s: %s", var, found)
            return found

    found = _verified_elpa_prefix(os.environ.get("EBROOTELPA"))
    if found:
        logger.debug("ELPA found via EBROOTELPA: %s", found)
        return found

    found = _from_pkg_config()
    if found:
        logger.debug("ELPA found via pkg-config: %s", found)
        return found

    found = _from_linker_paths()
    if found:
        logger.debug("ELPA found via linker search path: %s", found)
        return found

    return None


def find_cp2k_data_dir() -> Optional[str]:
    """Detect CP2K basis set / data directory.

    Resolution order:
      1. ``CP2K_DATA_DIR`` environment variable
      2. Check ``/usr/share/cp2k`` (Debian/Ubuntu convention)
      3. None if not found

    Returns:
        Absolute path to CP2K data directory, or None.
    """
    env = os.environ.get("CP2K_DATA_DIR")
    if env and Path(env).is_dir():
        return str(Path(env).resolve())

    debian = Path("/usr/share/cp2k")
    if debian.is_dir():
        return str(debian)

    return None
