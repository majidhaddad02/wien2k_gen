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

import json
import time
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Literal, TypedDict
from dataclasses import dataclass, field, asdict, is_dataclass, fields
from enum import Enum

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
    """Parallel execution strategies recognized by optimizer & backends."""
    MPI = "mpi"
    HYBRID = "hybrid"
    KPOINT = "kpoint"
    SERIAL = "serial"


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceSuggestion:
    """Optimizer output containing recommended parallel allocation."""
    mode: ExecutionMode = ExecutionMode.HYBRID
    recommended_total_cores: int = 1
    omp_threads_per_rank: int = 1
    mpi_ranks_per_node: int = 1
    cores_per_node: List[int] = field(default_factory=list)
    vector_split_active: bool = False
    warnings: List[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 1.0
    estimated_memory_gb: Optional[float] = None
    lapw0_cfg: StageConfig = field(default_factory=StageConfig)
    lapw1_cfg: StageConfig = field(default_factory=StageConfig)
    lapw2_cfg: StageConfig = field(default_factory=StageConfig)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value  # Convert Enum to string for JSON
        return d

    def validate_memory(self, topo_memory_limit_mb: Optional[float] = None) -> List[str]:
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


@dataclass(frozen=True)
class TopologyData:
    """Hardware & scheduler topology container. Immutable for thread-safety."""
    nodes: List[str] = field(default_factory=list)
    cores_per_node: List[int] = field(default_factory=list)
    env_type: str = "local"
    total_cores: int = 0
    scheduler_hints: Dict[str, Any] = field(default_factory=dict)
    heterogeneous: bool = False
    memory_per_node: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.heterogeneous and self.cores_per_node:
            is_het = len(set(self.cores_per_node)) > 1
            object.__setattr__(self, "heterogeneous", is_het)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopologyData":
        """Reconstruct from dictionary with safe fallbacks."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class PipelineResult:
    """Outcome of the central configuration generation pipeline."""
    success: bool = False
    config_path: Optional[str] = None
    config_content: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dry_run_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return self.success and not self.validation_errors

    def to_dict(self) -> Dict[str, Any]:
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

    def to_slurm_directives(self) -> Dict[str, Any]:
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
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


def deep_merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
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
    except ValueError:
        raise ValueError(f"Invalid '{field_name}': expected {enum_cls.__name__}, got '{value}'")


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "BackendCode",
    "ExecutionMode",
    "JobStatus",
    "OptimizationTarget",
    "StageConfig",
    "ResourceSuggestion",
    "TopologyData",
    "PipelineResult",
    "SubmissionConfig",
    "BenchmarkResult",
    "to_serializable",
    "deep_merge_configs",
    "validate_enum_field",
]