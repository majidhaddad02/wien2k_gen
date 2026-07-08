"""
Centralized Exception Hierarchy & Error Context Manager for Wien2kGen.
Provides structured, HPC-aware error handling with machine-readable codes,
actionable troubleshooting hints, severity classification, and seamless
integration with logging, CLI, TUI, and pipeline diagnostics.

Key Architecture Features:
• Hierarchical exception design matching HPC workflow domains
• Rich context metadata: error_code, severity, domain, hint, original_cause, context dict
• Automatic JSON serialization for structured logging & UI error panels
• Severity-to-logging-level mapping & Rich/Textual color hints
• Thread-safe error propagation & exception chaining (cause)
• Zero-dependency design: pure stdlib typing & traceback handling
• Comprehensive English documentation, type hints, and production-grade resilience

All documentation and inline comments are in English per project standards.
"""

import json
import logging
import traceback
from enum import Enum

# Avoid circular import: use TYPE_CHECKING and lazy import for get_logger
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    pass


# =============================================================================
# Severity & Domain Constants
# =============================================================================

class ErrorSeverity(str, Enum):
    """Standardized severity levels for UI coloring & logging routing."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorDomain(str, Enum):
    """Functional domains for error routing & diagnostic tagging."""
    TOPOLOGY = "topology"
    SCHEDULER = "scheduler"
    CONFIGURATION = "configuration"
    RUNTIME = "runtime"
    BACKEND = "backend"
    PIPELINE = "pipeline"
    FILESYSTEM = "filesystem"
    UNKNOWN = "unknown"


# =============================================================================
# Base Exception with Rich Context
# =============================================================================

class Wien2kGenError(Exception):
    """
    Base exception for all Wien2kGen errors.
    Carries structured metadata for logging, UI display, and automated diagnostics.
    Supports exception chaining via `__cause__` and flexible context injection.
    """
    error_code: str = "W2K_000"
    exit_code: int = 1

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: Union[ErrorSeverity, str] = ErrorSeverity.ERROR,
        domain: Union[ErrorDomain, str] = ErrorDomain.UNKNOWN,
        hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        recoverable: bool = False
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.error_code
        self.severity = ErrorSeverity(severity)
        self.domain = ErrorDomain(domain)
        self.hint = hint or self._generate_default_hint()
        self.context = context or {}
        self.original_exception = original_exception
        self.recoverable = recoverable
        self.traceback_str = self._capture_traceback()

        # Preserve exception chain
        if original_exception is not None:
            self.__cause__ = original_exception

    def _generate_default_hint(self) -> str:
        """Generate context-aware troubleshooting hint based on domain."""
        hints = {
            ErrorDomain.TOPOLOGY: "Check lscpu, numactl, and scheduler environment variables.",
            ErrorDomain.SCHEDULER: "Verify squeue/mpirun availability and partition limits.",
            ErrorDomain.CONFIGURATION: "Review .machines/INCAR syntax and divisibility constraints.",
            ErrorDomain.RUNTIME: "Check memory limits, MPI libraries, and scratch permissions.",
            ErrorDomain.BACKEND: "Ensure WIENROOT/QE_BIN/VASP_ROOT is correctly set.",
            ErrorDomain.FILESYSTEM: "Verify read/write access to SCRATCH and working directory.",
        }
        return hints.get(self.domain, "Consult documentation or run wien2k_gen --diagnostics")

    def _capture_traceback(self) -> Optional[str]:
        """Format traceback if original exception exists."""
        if self.original_exception:
            return "".join(traceback.format_exception(
                type(self.original_exception),
                self.original_exception,
                self.original_exception.__traceback__
            ))
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dictionary for logging/UI."""
        return {
            "error_code": self.error_code,
            "severity": self.severity.value,
            "domain": self.domain.value,
            "message": self.message,
            "hint": self.hint,
            "recoverable": self.recoverable,
            "context": self.context,
            "original_type": type(self.original_exception).__name__ if self.original_exception else None,
            "traceback": self.traceback_str
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to formatted JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=indent, ensure_ascii=False)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} code='{self.error_code}' "
            f"severity='{self.severity.value}' domain='{self.domain.value}' "
            f"recoverable={self.recoverable}>"
        )


# =============================================================================
# Domain-Specific Exception Hierarchy
# =============================================================================

class TopologyError(Wien2kGenError):
    """Hardware or scheduler topology detection/alignment failures."""
    error_code = "W2K_TOP_001"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, domain=ErrorDomain.TOPOLOGY, **kwargs)


class DetectionFailedError(TopologyError):
    """Raised when lscpu/numactl/scheduler parsing fails."""
    error_code = "W2K_TOP_010"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, hint="Ensure lscpu, numactl, and sinfo are in PATH.", **kwargs)


class InvalidTopologyError(TopologyError):
    """Raised when detected cores/nodes are inconsistent or negative."""
    error_code = "W2K_TOP_020"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class SchedulerError(Wien2kGenError):
    """SLURM/PBS/LSF communication & job dispatch failures."""
    error_code = "W2K_SCH_001"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, domain=ErrorDomain.SCHEDULER, **kwargs)


class SubmissionFailedError(SchedulerError):
    """Raised when sbatch/qsub/jsrun rejects the job."""
    error_code = "W2K_SCH_010"

    def __init__(self, message: str, job_id: Optional[int] = None, **kwargs: Any) -> None:
        ctx = kwargs.get("context", {})
        if job_id is not None:
            ctx["job_id"] = job_id
        kwargs["context"] = ctx
        super().__init__(message, **kwargs)


class PreemptionError(SchedulerError):
    """Raised when job is preempted or hits walltime limit."""
    error_code = "W2K_SCH_020"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            hint="Adjust walltime, enable checkpointing, or use non-preemptable QoS.",
            **kwargs
        )


class ConfigurationError(Wien2kGenError):
    """Parallel config generation, validation, or syntax failures."""
    error_code = "W2K_CFG_001"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, domain=ErrorDomain.CONFIGURATION, **kwargs)


class ValidationError(ConfigurationError):
    """Raised when .machines/INCAR/divisibility checks fail."""
    error_code = "W2K_CFG_010"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, hint="Check KPAR/NCORE/omp divisibility and core limits.", **kwargs)


class GenerationError(ConfigurationError):
    """Raised when backend input generation throws an exception."""
    error_code = "W2K_CFG_020"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class HPCRuntimeError(Wien2kGenError):
    """Execution-time failures on compute nodes or login nodes."""
    error_code = "W2K_RUN_001"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, domain=ErrorDomain.RUNTIME, **kwargs)


class MPIError(HPCRuntimeError):
    """MPI communication, rank mismatch, or library loading failures."""
    error_code = "W2K_RUN_010"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, hint="Verify MPI vendor, UCX/OFI settings, and LD_LIBRARY_PATH.", **kwargs)


class ScratchError(HPCRuntimeError):
    """Scratch creation, staging, or permission failures."""
    error_code = "W2K_RUN_020"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, hint="Check /dev/shm, $SCRATCH permissions, and NFS quota.", **kwargs)


class TimeoutError(HPCRuntimeError):
    """Raised when job or subprocess exceeds configured timeout."""
    error_code = "W2K_RUN_030"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, severity=ErrorSeverity.WARNING, **kwargs)


class BackendError(Wien2kGenError):
    """DFT backend-specific parsing, binary, or compatibility failures."""
    error_code = "W2K_BKD_001"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, domain=ErrorDomain.BACKEND, **kwargs)


class MissingInputError(BackendError):
    """Raised when required input files (case.struct, INCAR, *.in) are missing."""
    error_code = "W2K_BKD_010"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class ParsingError(BackendError):
    """Raised when output/log parsing fails due to unexpected format."""
    error_code = "W2K_BKD_020"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, hint="Check WIEN2k/QE/VASP version compatibility.", **kwargs)


# =============================================================================
# Utility Functions for Error Handling & Routing
# =============================================================================

def raise_with_context(
    exc_type: type,
    message: str,
    original: Optional[Exception] = None,
    hint: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    **context_kwargs: Any
) -> None:
    """
    Raise a structured exception with full context & automatic traceback capture.
    Thread-safe and compatible with async workers.
    """
    raise exc_type(
        message=message,
        original_exception=original,
        hint=hint,
        severity=severity,
        context=context_kwargs or {}
    )


def format_error_for_ui(error: Wien2kGenError) -> str:
    """
    Format exception for Rich/Textual UI panels with color-coded severity.
    """
    color_map = {
        ErrorSeverity.INFO: "blue",
        ErrorSeverity.WARNING: "yellow",
        ErrorSeverity.ERROR: "red",
        ErrorSeverity.CRITICAL: "bold red"
    }
    color = color_map.get(error.severity, "white")
    lines = [f"[{color} bold]{error.error_code}[/] {error.message}"]
    if error.hint:
        lines.append(f"[dim]💡 Hint: {error.hint}[/]")
    if error.recoverable:
        lines.append("[dim]⚠️  This error may be recoverable. Check logs.[/]")
    return "\n".join(lines)


def is_wien2k_error(exc: Exception) -> bool:
    """Check if exception belongs to the project's structured hierarchy."""
    return isinstance(exc, Wien2kGenError)


def log_exception_structured(exc: Exception, level: int = logging.ERROR) -> None:
    """
    Log exception with structured JSON context for centralized log aggregation.
    """
    # Lazy import to avoid circular dependency
    from .logging_config import get_logger
    logger = get_logger(__name__)
    
    if is_wien2k_error(exc):
        logger.log(level, "Structured error: %s", exc.to_dict())  # type: ignore[attr-defined]
    else:
        logger.log(level, "Unhandled exception: %s", traceback.format_exc())


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    # Enums
    "ErrorSeverity",
    "ErrorDomain",
    # Base Exception
    "Wien2kGenError",
    # Topology
    "TopologyError",
    "DetectionFailedError",
    "InvalidTopologyError",
    # Scheduler
    "SchedulerError",
    "SubmissionFailedError",
    "PreemptionError",
    # Configuration
    "ConfigurationError",
    "ValidationError",
    "GenerationError",
    # Runtime
    "HPCRuntimeError",
    "MPIError",
    "ScratchError",
    "TimeoutError",
    # Backend
    "BackendError",
    "MissingInputError",
    "ParsingError",
    # Utilities
    "raise_with_context",
    "format_error_for_ui",
    "is_wien2k_error",
    "log_exception_structured",
]