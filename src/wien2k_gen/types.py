"""
Central Type Definitions & Data Contracts for Wien2kGen.
Single source of truth for pipeline data, ensuring type safety, JSON serialization,
and seamless integration across core, optimizer, backends, UI, benchmark, and CLI modules.

Key Architecture Features:
• Unified Enum types for backend codes, execution modes, job states, and optimization targets
• Strict dataclass configurations with automatic dict/JSON conversion
• Immutable topology & submission configs to prevent race conditions
• Robust serialization utilities handling Path, datetime, Enum, and nested structures
• Deep-merge & validation helpers for multi-layer configuration resolution
• Zero circular imports: hardware/scheduler limits passed explicitly
• Comprehensive English documentation, type hints, and HPC-grade resilience patterns

All documentation and inline comments are in English per project standards.
"""

import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

# =============================================================================
# Domain Enums (JSON-safe via string inheritance)
# =============================================================================

class BackendCode(str, Enum):
    """Supported DFT backend identifiers."""
    WIEN2K = "wien2k"
    QUANTUM_ESPRESSO = "qe"
    VASP = "vasp"
    CP2K = "cp2k"


class ExecutionMode(str, Enum):
    """Parallel execution strategies recognized by optimizer & backends.
    
    Reference: Blaha, P. et al. (2020). WIEN2k Usersguide, Section 4.5.
    """
    MPI = "mpi"
    HYBRID = "hybrid"
    KPOINT = "kpoint"
    SERIAL = "serial"
    FINE_GRAIN = "fine_grain"


class CalculationType(str, Enum):
    """WIEN2k calculation types with associated run_lapw flags.
    
    Reference: WIEN2k Usersguide, Section 4.1-4.4.
    """
    SCF = "scf"                   # Standard SCF: run_lapw -p
    SPIN_POLARIZED = "spin"       # Spin-polarized: runsp_lapw -p
    SPIN_ORBIT = "soc"            # + Spin-orbit coupling: run_lapw -p -so
    SPIN_POLARIZED_SOC = "spin_soc"  # Spin-polarized + SOC: runsp_lapw -p -so
    LDA_U = "ldau"               # LDA/GGA+U: run_lapw -p -orbc
    HYBRID_FUNC = "hybrid"       # Hybrid functional: run_lapw -p -hf
    FORCES = "forces"            # Forces optimization: run_lapw -p -fc
    EECE = "eece"                # Onsite exact exchange: run_lapw -p -eece


class Wien2kVersion(str, Enum):
    """WIEN2k versions with known parallelization characteristics."""
    V19 = "19"
    V21 = "21"
    V23 = "23"
    V24 = "24"
    UNKNOWN = "unknown"


class JobStatus(str, Enum):
    """Lifecycle states for SLURM/scheduler job tracking."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"


class OptimizationTarget(str, Enum):
    """Objective functions for resource advisor & profiler tuning."""
    TIME = "time"
    MEMORY = "memory"
    BALANCED = "balanced"
    COST = "cost"


# =============================================================================
# Core Data Models
# =============================================================================

@dataclass
class StageConfig:
    """Configuration for a specific execution stage (lapw0/1/2)."""
    max_ranks: int = 1
    omp_threads: int = 1
    memory_per_rank_gb: float = 2.0
    io_strategy: Literal["local", "split", "collective"] = "local"
    vector_split_factor: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceSuggestion:
    """Optimizer output containing recommended parallel allocation."""
    mode: ExecutionMode = ExecutionMode.HYBRID
    recommended_total_cores: int = 1
    omp_threads_per_rank: int = 1
    mpi_ranks_per_node: int = 1
    cores_per_node: list[int] = field(default_factory=list)
    vector_split_active: bool = False
    warnings: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 1.0
    estimated_memory_gb: Optional[float] = None
    lapw0_cfg: StageConfig = field(default_factory=StageConfig)
    lapw1_cfg: StageConfig = field(default_factory=StageConfig)
    lapw2_cfg: StageConfig = field(default_factory=StageConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value  # Convert Enum to string for JSON
        return d

    def validate_memory(self, topo_memory_limit_mb: Optional[float] = None) -> list[str]:
        """
        Validate memory allocation against scheduler/hardware limits.
        Note: Accepts limit as argument to avoid circular imports with core.hardware
        """
        errors = []
        if self.estimated_memory_gb and topo_memory_limit_mb:
            mem_per_core_mb = (self.estimated_memory_gb * 1024) / max(1, self.recommended_total_cores)
            if mem_per_core_mb > topo_memory_limit_mb * 0.9:
                errors.append(
                    f"Memory per core ({mem_per_core_mb:.0f} MB) exceeds scheduler limit ({topo_memory_limit_mb} MB)"
                )
        return errors


@dataclass
class Wien2kFlags:
    """WIEN2k calculation flags detected from input files.
    
    Determines the correct run_lapw command and parallelization strategy.
    Reference: Blaha, P. et al. (2020). WIEN2k Usersguide, Sections 4.1-4.4.
    """
    is_spin_polarized: bool = False
    is_soc: bool = False
    is_lda_u: bool = False
    is_hybrid: bool = False
    is_eece: bool = False
    has_forces: bool = False
    wien2k_version: str = "unknown"

    def get_calculation_type(self) -> CalculationType:
        if self.is_spin_polarized and self.is_soc:
            return CalculationType.SPIN_POLARIZED_SOC
        if self.is_spin_polarized:
            return CalculationType.SPIN_POLARIZED
        if self.is_soc:
            return CalculationType.SPIN_ORBIT
        if self.is_hybrid:
            return CalculationType.HYBRID_FUNC
        if self.is_lda_u:
            return CalculationType.LDA_U
        if self.is_eece:
            return CalculationType.EECE
        if self.has_forces:
            return CalculationType.FORCES
        return CalculationType.SCF

    def get_execution_command(self) -> str:
        """Return the correct run_lapw command with all required flags."""
        calc_type = self.get_calculation_type()
        if calc_type in (CalculationType.SPIN_POLARIZED, CalculationType.SPIN_POLARIZED_SOC):
            cmd = "runsp_lapw"
        else:
            cmd = "run_lapw"

        flags = ["-p"]
        if calc_type in (CalculationType.SPIN_ORBIT, CalculationType.SPIN_POLARIZED_SOC):
            flags.append("-so")
        if self.is_lda_u:
            flags.append("-orbc")
        if self.is_hybrid:
            flags.append("-hf")
        if self.is_eece:
            flags.append("-eece")
        if self.has_forces:
            flags.append("-fc")

        return " ".join([cmd, *flags])


@dataclass(frozen=True)
class TopologyData:
    """Hardware & scheduler topology container. Immutable for thread-safety."""
    nodes: list[str] = field(default_factory=list)
    cores_per_node: list[int] = field(default_factory=list)
    env_type: str = "local"
    total_cores: int = 0
    scheduler_hints: dict[str, Any] = field(default_factory=dict)
    heterogeneous: bool = False
    memory_per_node: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.heterogeneous and self.cores_per_node:
            is_het = len(set(self.cores_per_node)) > 1
            object.__setattr__(self, "heterogeneous", is_het)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyData":
        """Reconstruct from dictionary with safe fallbacks."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class PipelineResult:
    """Outcome of the central configuration generation pipeline."""
    success: bool = False
    config_path: Optional[str] = None
    config_content: Optional[str] = None
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dry_run_content: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return self.success and not self.validation_errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _auto_default_mem() -> str:
    """Return a sensible default memory string based on 80% of system RAM."""
    try:
        from .core.hardware import get_total_mem_kb
        total_gb = get_total_mem_kb() / (1024 * 1024)
        mem_gb = int(total_gb * 0.8)
        if mem_gb < 1:
            mem_gb = 8
        return f"{mem_gb}G"
    except Exception:
        return "8G"


@dataclass(frozen=True)
class SubmissionConfig:
    """SLURM/job submission parameters. Frozen for consistency."""
    job_name: str = "wien2k_job"
    partition: str = ""
    nodes: int = 1
    ntasks: int = 0  # 0 = auto-calculate from topo
    cpus_per_task: int = 1
    mem_per_node: str = field(default_factory=lambda: _auto_default_mem())
    walltime: str = "24:00:00"
    dependency: str = ""
    qos: str = ""
    gres: str = ""
    account: str = ""
    preemption_grace_sec: int = 60
    dry_run: bool = False

    def to_slurm_directives(self) -> dict[str, Any]:
        """Convert to submit/slurm.py compatible directive dictionary."""
        return {k: v for k, v in asdict(self).items() if k not in ("dry_run", "preemption_grace_sec")}


@dataclass
class BenchmarkResult:
    """Unified result container for synthetic & real cluster benchmarks."""
    run_id: str = ""
    backend: BackendCode = BackendCode.WIEN2K
    mode: ExecutionMode = ExecutionMode.MPI
    total_cores: int = 0
    wall_time_sec: float = 0.0
    cpu_time_sec: float = 0.0
    efficiency_percent: float = 0.0
    bottleneck: str = ""
    converged: bool = False
    job_id: Optional[int] = None
    log_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["backend"] = self.backend.value
        d["mode"] = self.mode.value
        return d


# =============================================================================
# Serialization & Configuration Utilities
# =============================================================================

def to_serializable(obj: Any) -> Any:
    """
    Recursively convert complex objects to JSON-safe primitives.
    Handles Path, Enum, datetime, dataclass, TypedDict, and nested structures.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (Path, datetime)):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(i) for i in obj]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return to_serializable(obj.to_dict())
    return str(obj)


def deep_merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-merge two configuration dictionaries with last-write-wins semantics.
    Preserves nested structures and avoids shallow-copy mutation bugs.
    """
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def validate_enum_field(value: Any, enum_cls: type, field_name: str) -> Enum:
    """
    Validate & coerce string/input to target Enum type.
    Raises ValueError with actionable message on mismatch.
    """
    try:
        return value if isinstance(value, enum_cls) else enum_cls(value)
    except ValueError as e:
        raise ValueError(f"Invalid '{field_name}': expected {enum_cls.__name__}, got '{value}'") from e


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "BackendCode",
    "BenchmarkResult",
    "CalculationType",
    "ExecutionMode",
    "JobStatus",
    "OptimizationTarget",
    "PipelineResult",
    "ResourceSuggestion",
    "StageConfig",
    "SubmissionConfig",
    "TopologyData",
    "Wien2kFlags",
    "Wien2kVersion",
    "deep_merge_configs",
    "to_serializable",
    "validate_enum_field",
]