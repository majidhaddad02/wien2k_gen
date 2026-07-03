"""
Atomic File Write Utility Module.
Provides guaranteed atomic file write operations using `tempfile` and `os.replace`.
Designed for HPC environments where parallel jobs may read configuration files
while they are being generated, preventing partial read errors or crashes.

Key features:
• Atomic rename on the same filesystem (POSIX standard).
• Explicit `fsync` calls to flush data to disk before renaming.
• Preservation of file permissions (mode).
• Robust error handling with automatic temporary file cleanup on failure.
• Support for both text (str) and binary (bytes) content.
• Thread-safe and process-safe file creation.
• Ensures target directory exists before attempting write.

All documentation and inline comments are in English per project standards.
"""

import os
import tempfile
import logging
import stat
from pathlib import Path
from typing import Union, AnyStr, Optional, List

from ..logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Core Atomic Write Logic
# =============================================================================

def atomic_write(
    path: Union[str, Path],
    content: Union[str, bytes],
    mode: int = 0o644,
    encoding: str = "utf-8",
    ensure_permissions: bool = True
) -> bool:
    """
    Atomically write content to a file by writing to a temporary file first
    and then renaming it. This ensures that no other process ever reads a
    partially written file.

    Args:
        path: Destination file path.
        content: Data to write (str or bytes).
        mode: File permissions (e.g., 0o644).
        encoding: Text encoding (used if content is str).
        ensure_permissions: If True, enforce 'mode' on the new file.

    Returns:
        True if write was successful, False otherwise.

    Raises:
        OSError: If disk is full, permission denied, or filesystem error occurs.
    """
    target_path = Path(path).resolve()
    target_dir = target_path.parent

    # Ensure target directory exists
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {target_dir}: {e}")
        raise

    # Create a temporary file in the same directory to guarantee same filesystem
    # for the atomic rename operation. Prefix with '.' to hide it initially.
    fd = None
    tmp_path = None
    
    try:
        # tempfile.mkstemp returns (fd, abs_path)
        # dir=target_dir ensures the temp file is on the same mount point
        # as the target file, which is required for os.replace() to be atomic.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target_dir),
            prefix=f".tmp_{target_path.name}.",
            suffix=".tmp"
        )

        # Open the file descriptor for writing
        # Use binary mode for precise control over fsync
        with os.fdopen(fd, "wb") as tmp_file:
            fd = None  # fdopen takes ownership of fd

            # Write content
            if isinstance(content, str):
                tmp_file.write(content.encode(encoding)) 
            else:
                tmp_file.write(content)

            # Flush user-space buffers
            tmp_file.flush()

            # Flush kernel buffers to disk (critical for power failures / crashes)
            os.fsync(tmp_file.fileno())

        # Set permissions before rename if requested
        if ensure_permissions:
            os.chmod(tmp_path, mode)

        # Atomic rename
        # os.replace is atomic on POSIX systems if both paths are on the same filesystem.
        # It will overwrite the target if it exists.
        os.replace(tmp_path, str(target_path))
        tmp_path = None  # Mark as successful so finally block doesn't clean it

        logger.debug(f"Atomically wrote {len(content)} bytes to {target_path}")
        return True

    except Exception as e:
        logger.error(f"Atomic write failed for {target_path}: {e}")
        raise
        
    finally:
        # Cleanup temporary file if rename didn't happen
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# =============================================================================
# Convenience Wrappers
# =============================================================================

def atomic_write_json(path: Union[str, Path], data: dict, indent: int = 2) -> bool:
    """
    Convenience function to write JSON data atomically.
    """
    import json
    return atomic_write(
        path, 
        json.dumps(data, indent=indent) + "\n", 
        mode=0o644
    )


def atomic_write_list(path: Union[str, Path], lines: List[str], newline: str = "\n") -> bool:
    """
    Convenience function to write a list of lines atomically.
    """
    content = newline.join(lines)
    if content and not content.endswith(newline):
        content += newline
    return atomic_write(path, content, mode=0o644)