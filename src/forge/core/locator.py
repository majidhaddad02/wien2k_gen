"""
Centralized external dependency locator for FORGE.
Auto-detects WIEN2k, ELPA, CP2K installation paths at runtime
using PATH scanning and environment variables -- no hardcoded paths.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


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


def find_elpa_dir() -> Optional[str]:
    """Detect ELPA library installation directory.

    Resolution order:
      1. ``ELPA_HOME`` environment variable
      2. Try common module paths (``$MODULEPATH`` scan)
      3. None if not found

    Returns:
        Absolute path to ELPA installation, or None.
    """
    env = os.environ.get("ELPA_HOME")
    if env and Path(env).is_dir():
        return str(Path(env).resolve())

    for root in os.environ.get("MODULEPATH", "").split(":"):
        candidate = Path(root) / "ELPA"
        if candidate.is_dir():
            return str(candidate)

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
