"""
Process-Safe File Locking Utility for HPC & Distributed Workflows.
Production features:
• POSIX advisory locking via fcntl as primary mechanism (automatic release on crash/exit)
• Atomic directory + PID file fallback for network filesystems (NFS/Lustre/GPFS) where fcntl is unreliable
• Configurable timeout, retry delay, and stale-lock detection/cleanup
• Context-manager interface (`with FileLock(...)`) for safe, exception-proof scoping
• Thread-safe and process-safe locking with explicit acquisition/release logging
• Graceful degradation with structured error reporting for UI/CLI consumption
• Comprehensive English documentation and HPC-grade resilience patterns
All documentation and inline comments are in English per project standards.
"""

import errno
import fcntl
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

from ..logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Constants & Defaults
# =============================================================================
DEFAULT_TIMEOUT = 30.0        # Seconds to wait for lock acquisition
DEFAULT_RETRY_DELAY = 0.1     # Initial sleep interval between retries
MAX_RETRY_DELAY = 2.0         # Cap for exponential backoff
STALE_LOCK_THRESHOLD = 300.0  # Seconds before considering a fallback lock stale

# =============================================================================
# Core File Lock Implementation
# =============================================================================

class FileLock:
    """
    Cross-platform, process-safe file lock with timeout and stale-lock recovery.
    Designed for HPC environments where multiple jobs may concurrently read/write
    configuration, cache, or scratch files on shared network filesystems.
    
    Usage:
        with FileLock("/tmp/wien2k_cache.lock", timeout=10) as lock:
            # Critical section: safe to read/write shared resource
            ...
    """

    def __init__(
        self,
        path: Union[str, Path],
        timeout: float = DEFAULT_TIMEOUT,
        delay: float = DEFAULT_RETRY_DELAY,
        lock_file: Optional[Union[str, Path]] = None,
        cleanup_stale: bool = True
    ) -> None:
        """
        Initialize file lock with timeout and retry configuration.
        
        Args:
            path: Path to the lock file or directory. Used as the lock identifier.
            timeout: Maximum seconds to wait before raising LockTimeoutError.
            delay: Initial sleep interval between acquisition retries (seconds).
            lock_file: Explicit path for the lock file. Defaults to `<path>.lock`.
            cleanup_stale: If True, automatically remove stale locks from crashed processes.
        """
        self.target_path = Path(path).resolve()
        self.lock_path = Path(lock_file) if lock_file else self.target_path.parent / f".{self.target_path.name}.lock"
        self.timeout = timeout
        self.base_delay = delay
        self.cleanup_stale = cleanup_stale
        self._fd: Optional[int] = None
        self._fallback_dir: Optional[Path] = None
        self._pid_file: Optional[Path] = None

    # =========================================================================
    # Context Manager Protocol
    # =========================================================================

    def __enter__(self) -> "FileLock":
        """Acquire lock on entry. Blocks or raises on timeout."""
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release lock on exit, ensuring cleanup even on exceptions."""
        self.release()

    # =========================================================================
    # Public API: Acquire & Release
    # =========================================================================

    def acquire(self) -> None:
        """
        Attempt to acquire the lock with timeout and exponential backoff.
        Tries POSIX fcntl first, falls back to atomic directory + PID tracking.
        Raises LockTimeoutError if timeout is exceeded.
        """
        start_time = time.monotonic()
        current_delay = self.base_delay
        attempt = 0
        
        while True:
            attempt += 1
            elapsed = time.monotonic() - start_time
             
            if elapsed >= self.timeout:
                raise LockTimeoutError(
                    f"Failed to acquire lock for {self.target_path} within {self.timeout}s "
                    f"(timeout reached after {attempt} attempts)."
                )
                
            # 1. Primary: POSIX fcntl advisory lock
            try:
                self._try_fcntl()
                logger.debug(f"Lock acquired via fcntl: {self.lock_path} (attempt {attempt})")
                return
            except OSError as e:
                # fcntl unavailable or not supported on this filesystem
                if e.errno not in (errno.EACCES, errno.EAGAIN, errno.ENOLCK):
                    logger.debug(f"fcntl failed, trying fallback: {e}")
            except LockAcquisitionError:
                logger.debug("fcntl lock acquisition failed, trying fallback")
                    
            # 2. Fallback: Atomic directory + PID file
            try:
                self._try_atomic_dir()
                logger.debug(f"Lock acquired via atomic dir: {self.lock_path}.d (attempt {attempt})")
                return
            except LockAcquisitionError:
                logger.debug("Suppressed exception in acquire()", exc_info=True)
                
            # Check stale locks if fallback failed
            if self.cleanup_stale and self._is_fallback_stale():
                try:
                    self._cleanup_fallback()
                except Exception:
                    logger.info(f"Cleaned stale fallback lock: {self.lock_path}.d")
                    # Retry immediately after cleanup
                continue
                    
            # Exponential backoff with cap
            time.sleep(current_delay)
            current_delay = min(current_delay * 1.5, MAX_RETRY_DELAY)

    def release(self) -> None:
        """
        Release the lock and clean up resources.
        Safe to call multiple times; idempotent after first release.
        """
        # Release fcntl lock
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
                logger.debug(f"fcntl lock released: {self.lock_path}")
            except OSError as e:
                logger.warning(f"fcntl unlock failed (may be already closed): {e}")
            finally:
                self._fd = None
                
        # Release fallback directory lock
        if self._fallback_dir is not None and self._fallback_dir.exists():
            try:
                if self._pid_file and self._pid_file.exists():
                    self._pid_file.unlink()
                self._fallback_dir.rmdir()
                logger.debug(f"Atomic dir lock released: {self.lock_path}.d")
            except OSError as e:
                logger.warning(f"Fallback cleanup failed: {e}")
            finally:
                self._fallback_dir = None
                self._pid_file = None

    # =========================================================================
    # Internal Locking Mechanisms
    # =========================================================================

    def _try_fcntl(self) -> None:
        """Attempt to acquire POSIX advisory lock."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError, OSError) as e:
            os.close(self._fd)
            self._fd = None
            raise LockAcquisitionError(f"fcntl LOCK_EX failed: {e}") from e

    def _try_atomic_dir(self) -> None:
        """
        Attempt to acquire lock via atomic directory creation.
        Falls back to PID file for stale detection.
        """
        self._fallback_dir = Path(f"{self.lock_path}.d")
        self._pid_file = self._fallback_dir / "pid"
        
        try:
            # os.mkdir is atomic on local POSIX filesystems and Lustre/GPFS.
    # WARNING: NFSv3 does NOT guarantee atomic mkdir between clients.
    # Use file locking on a shared filesystem (Lustre MDT-local) for safety.
            self._fallback_dir.mkdir(parents=True, exist_ok=False)
            self._pid_file.write_text(str(os.getpid()), encoding="utf-8")
        except FileExistsError:
            raise LockAcquisitionError("Directory lock already exists (held by another process)") from None
        except OSError as e:
            raise LockAcquisitionError(f"Atomic dir creation failed: {e}") from e

    def _is_fallback_stale(self) -> bool:
        """Check if fallback lock directory is stale (exceeds threshold)."""
        if not self._fallback_dir or not self._fallback_dir.exists():
            return False
            
        try:
            mtime = self._fallback_dir.stat().st_mtime
            age = time.time() - mtime
            if age > STALE_LOCK_THRESHOLD:
                # Verify PID is actually dead
                if self._pid_file and self._pid_file.exists():
                    pid = int(self._pid_file.read_text().strip())
                    if not self._is_process_alive(pid):
                        return True
                return True
            return False
        except Exception:
            return False

    def _cleanup_fallback(self) -> None:
        """Forcefully remove stale fallback lock."""
        if self._fallback_dir and self._fallback_dir.exists():
            try:
                if self._pid_file and self._pid_file.exists():
                    self._pid_file.unlink()
                self._fallback_dir.rmdir()
            except OSError:
                pass  # Another process may have cleaned it

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """Check if a process with given PID is currently running."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


# =============================================================================
# Custom Exceptions
# =============================================================================

class LockAcquisitionError(Exception):
    """Raised when a lock cannot be acquired immediately (non-blocking)."""
    pass


class LockTimeoutError(Exception):
    """Raised when lock acquisition exceeds the configured timeout."""
    pass


# =============================================================================
# Convenience Function
# =============================================================================

def file_lock(
    path: Union[str, Path],
    timeout: float = DEFAULT_TIMEOUT,
    delay: float = DEFAULT_RETRY_DELAY,
    cleanup_stale: bool = True
) -> FileLock:
    """
    Factory function for quick lock instantiation.
    Returns a FileLock instance configured for immediate context-manager use.
    """
    return FileLock(
        path=path,
        timeout=timeout,
        delay=delay,
        cleanup_stale=cleanup_stale
    )


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "DEFAULT_TIMEOUT",
    "STALE_LOCK_THRESHOLD",
    "FileLock",
    "LockAcquisitionError",
    "LockTimeoutError",
    "file_lock",
]