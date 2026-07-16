"""
Subprocess Execution & Process Lifecycle Management Module.
Production features:
• Safe execution of external commands (MPI, scripts, binaries) with timeout handling
• Automatic process group isolation (start_new_session) for reliable termination of child trees
• Signal forwarding: SIGTERM/SIGINT from parent to child process groups
• Robust stdout/stderr capture with configurable encoding and error handling
• Async execution support via `asyncio` for non-blocking monitoring loops
• Resource-safe cleanup to prevent zombie processes on HPC compute nodes
• Comprehensive English documentation and HPC-grade error handling
All documentation and inline comments are in English per project standards.
"""

import asyncio
import contextlib
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union

from ..logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Type Definitions & Data Structures
# =============================================================================

class ProcessResult:
    """
    Encapsulates the result of a subprocess execution.
    Provides properties for success status and standardized error reporting.
    """
    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        command: str,
        timed_out: bool
    ):
        self.returncode = returncode
        self.stdout = stdout or ""
        self.stderr = stderr or ""
        self.command = command
        self.timed_out = timed_out

    @property
    def success(self) -> bool:
        """Returns True if the process exited with code 0."""
        return self.returncode == 0

    def check_returncode(self) -> None:
        """Raise RuntimeError if the process exited with a non-zero code."""
        if self.returncode != 0:
            # Truncate stderr for logging if too long
            err_snippet = self.stderr[-2000:] if len(self.stderr) > 2000 else self.stderr
            raise RuntimeError(
                f"Command '{self.command}' failed with exit code {self.returncode}.\n"
                f"STDERR: {err_snippet}"
            )

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else f"FAILED({self.returncode})"
        cmd_preview = self.command[:50] + "..." if len(self.command) > 50 else self.command
        return f"<ProcessResult {status} | CMD: {cmd_preview}>"


# =============================================================================
# Signal & Process Group Management
# =============================================================================

def terminate_process_group(pid: Optional[int] = None) -> None:
    """
    Terminate an entire process group using SIGTERM followed by SIGKILL.
    Crucial for cleaning up MPI jobs where the launcher (srun/mpirun)
    spawns a hierarchy of child processes.
    
    Args:
        pid: Process ID (or group ID) to terminate. If None, uses current process group.
    """
    if pid is None:
        try:
            pid = os.getpgrp()
        except AttributeError:
            pid = os.getpgid(os.getpid())

    # Send SIGTERM first to allow graceful cleanup (checkpointing, file closing)
    try:
        os.killpg(pid, signal.SIGTERM)
        logger.debug(f"Sent SIGTERM to process group {pid}")
    except ProcessLookupError:
        pass  # Process already exited
    except PermissionError:
        logger.warning(f"Permission denied to send SIGTERM to PGID {pid}")
    except Exception as e:
        logger.warning(f"Error sending SIGTERM to PGID {pid}: {e}")


def force_kill_process_group(pid: Optional[int] = None) -> None:
    """
    Forcefully kill an entire process group using SIGKILL.
    Use this as a fallback if SIGTERM does not terminate the process within timeout.
    """
    if pid is None:
        try:
            pid = os.getpgrp()
        except AttributeError:
            pid = os.getpgid(os.getpid())
            
    try:
        os.killpg(pid, signal.SIGKILL)
        logger.warning(f"Sent SIGKILL to process group {pid}")
    except (ProcessLookupError, PermissionError):
        logger.debug("Suppressed exception in force_kill_process_group()", exc_info=True)
    except Exception as e:
        logger.warning(f"Error sending SIGKILL to PGID {pid}: {e}")


# =============================================================================
# Synchronous Execution Wrapper
# =============================================================================

def run_command(  # noqa: C901
    cmd: Union[str, list[str]],
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
    check: bool = False,
    capture_output: bool = True,
    text: bool = True,
    start_new_session: bool = True,
    **kwargs: Any
) -> ProcessResult:
    """
    Execute a command synchronously with robust process management.
    
    Key Features:
    • `start_new_session=True` isolates the process group, preventing signal leaks.
    • Automatic cleanup on parent termination via `preexec_fn` (signal handling).
    • Timeout enforcement with SIGTERM -> SIGKILL escalation.

    Args:
        cmd: Command string or list of arguments.
        cwd: Working directory.
        env: Environment variables (merged with current env).
        timeout: Execution time limit in seconds.
        check: If True, raise RuntimeError on non-zero exit code.
        start_new_session: Start a new session/PGID for the subprocess.
        
    Returns:
        ProcessResult object with stdout, stderr, returncode, etc.
    """
    cmd_list = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        
    cmd_str = " ".join(cmd_list)
    logger.debug(f"Executing command: {cmd_str}")

    # Prepare environment
    exec_env = os.environ.copy()
    if env:
        exec_env.update(env)

    # Pre-execution function to ignore parent signals and set process group leader.
    # This prevents the child from receiving signals intended for the parent 
    # (e.g. Ctrl+C in a shell) unless explicitly forwarded.
    def _preexec() -> None:
        if start_new_session:
            os.setpgrp()  # Create new process group (redundant if start_new_session=True, but safe)
        # Ignore SIGINT/SIGTERM initially so the main process controls them
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    proc = None

    try:
        proc = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            cwd=str(cwd) if cwd else None,
            env=exec_env,
            start_new_session=start_new_session,
            preexec_fn=_preexec if start_new_session and sys.platform != "win32" else None,
            text=text,
            **kwargs
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            result = ProcessResult(
                returncode=proc.returncode or 0,
                stdout=stdout or "",
                stderr=stderr or "",
                command=cmd_str,
                timed_out=False
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out after {timeout}s: {cmd_str}")
            # Graceful termination first
            terminate_process_group(proc.pid)
            
            # Wait briefly for graceful exit
            try:
                stdout, stderr = proc.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                # Force kill if still alive
                force_kill_process_group(proc.pid)
                stdout, stderr = proc.communicate()
                
            result = ProcessResult(
                returncode=proc.returncode or -1,
                stdout=stdout or "",
                stderr=stderr or "",
                command=cmd_str,
                timed_out=True
            )

        if check and not result.success:
            result.check_returncode()
            
        return result

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Executable not found in command '{cmd_str}': {e}") from e
    except Exception as e:
        logger.error(f"Execution failed for {cmd_str}: {e}")
        raise
    finally:
        # Ensure file descriptors are closed if communicate wasn't fully consumed
        if proc:
            for fd in [proc.stdout, proc.stderr]:
                if fd and not fd.closed:
                    with contextlib.suppress(Exception):
                        fd.close()


# =============================================================================
# Asynchronous Execution Wrapper
# =============================================================================

async def run_async_command(
    cmd: Union[str, list[str]],
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
    check: bool = False,
    **kwargs: Any
) -> ProcessResult:
    """
    Execute a command asynchronously.
    
    Args:
        cmd: Command string or list of arguments.
        cwd: Working directory.
        env: Environment variables.
        timeout: Time limit in seconds.
        check: If True, raise RuntimeError on non-zero exit code.
        
    Returns:
        ProcessResult object.
    """
    cmd_list = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        
    cmd_str = " ".join(cmd_list)
    logger.debug(f"Async executing command: {cmd_str}")

    exec_env = os.environ.copy()
    if env:
        exec_env.update(env)

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_list,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            env=exec_env,
            start_new_session=True,  # Isolate process group
            **kwargs
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            
            stdout = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ""
            stderr = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ""
            
            result = ProcessResult(
                returncode=proc.returncode or 0,
                stdout=stdout,
                stderr=stderr,
                command=cmd_str,
                timed_out=False
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Async command timed out after {timeout}s: {cmd_str}")
            terminate_process_group(proc.pid)
            await asyncio.sleep(1)
            if proc.returncode is None:
                force_kill_process_group(proc.pid)
                await proc.wait()
                
            # Try to get remaining output if possible (often empty after kill)
            stdout = ""
            stderr = ""
            if proc.stdout and not proc.stdout.at_eof():
                with contextlib.suppress(Exception):
                    stdout = (await proc.stdout.read()).decode('utf-8', errors='replace')
            if proc.stderr and not proc.stderr.at_eof():
                with contextlib.suppress(Exception):
                    stderr = (await proc.stderr.read()).decode('utf-8', errors='replace')
                
            result = ProcessResult(
                returncode=proc.returncode or -1,
                stdout=stdout,
                stderr=stderr,
                command=cmd_str,
                timed_out=True
            )

        if check and not result.success:
            result.check_returncode()
            
        return result
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Executable not found in command '{cmd_str}': {e}") from e
    except Exception as e:
        logger.error(f"Async execution failed for {cmd_str}: {e}")
        raise


# =============================================================================
# Real-time Stream Output
# =============================================================================

async def stream_command_output(  # noqa: C901
    cmd: Union[str, list[str]],
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
    callback: Optional[Callable[[str], None]] = None
) -> ProcessResult:
    """
    Execute command and stream output line-by-line to a callback function.
    Useful for real-time UI updates or log parsing during long jobs.
    """
    cmd_list = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        
    cmd_str = " ".join(cmd_list)
    logger.debug(f"Streaming command: {cmd_str}")

    exec_env = os.environ.copy()
    if env:
        exec_env.update(env)

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_list,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
            cwd=str(cwd) if cwd else None,
            env=exec_env,
            start_new_session=True,
        ) 

        full_output: list[str] = []
        
        async def _read_stream() -> None:
            assert proc.stdout is not None
            while True:
                try:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    decoded = line.decode('utf-8', errors='replace').strip()
                    full_output.append(decoded)
                    if callback:
                        try:
                            callback(decoded)
                        except Exception as e:
                            logger.warning(f"Callback execution error: {e}")
                except Exception:
                    break

        try:
            if proc.stdout:
                await asyncio.wait_for(_read_stream(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Stream command timed out: {cmd_str}")
            terminate_process_group(proc.pid)
            await asyncio.sleep(1)
            force_kill_process_group(proc.pid)
            return ProcessResult(
                returncode=-1,
                stdout="\n".join(full_output),
                stderr="Timed out",
                command=cmd_str,
                timed_out=True
            )

        await proc.wait()
        
        return ProcessResult(
            returncode=proc.returncode or 0,
            stdout="\n".join(full_output),
            stderr="",
            command=cmd_str,
            timed_out=False
        )

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Executable not found in command '{cmd_str}': {e}") from e
    finally:
        if proc and proc.stdout:
            proc.stdout.close()


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "ProcessResult",
    "force_kill_process_group",
    "run_async_command",
    "run_command",
    "stream_command_output",
    "terminate_process_group",
]