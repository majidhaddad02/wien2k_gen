"""
Unified Execution Pipeline for WIEN2k Configuration.
Orchestrates detection, advising, validation, building, and exporting.
Enhanced with rigorous preflight checks, HPC resource awareness, and robust error handling.

Key Improvements Applied:
- Fixed all whitespace corruption, syntax typos, and broken variable names.
- Corrected logger initialization to use __name__ instead of undefined 'name'.
- Prevented UnboundLocalError in exception handlers by initializing 'warnings' early.
- Fixed import comments and fallback definitions for type safety.
- Enhanced memory estimation with WIEN2k-specific heuristics.
- Integrated scratch space validation into preflight checks.
- All comments and documentation in English per project standards.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

# =============================================================================
# Type Definitions with Fallbacks (Ensures module works if types.py is missing)
# =============================================================================
try:
    from ..types import ProblemSize, ResourceSuggestion, PipelineResult
except ImportError:
    from dataclasses import dataclass, field
    from typing import List, Dict, Optional, Union, Any

    @dataclass
    class ProblemSize:
        """Lightweight representation of DFT problem scale."""
        n_atoms: int = 0
        n_kpoints: int = 0
        n_bands: int = 0
        is_spin_polarized: bool = False
        has_spin_orbit: bool = False
        memory_estimate_mb: float = 0.0

    @dataclass
    class ResourceSuggestion:
        """Structured recommendation for MPI/OpenMP hybrid configuration."""
        mode: str = "mpi"
        recommended_total_cores: int = 1
        omp_threads_per_rank: int = 1
        mpi_ranks_per_node: int = 1
        cores_per_node: List[int] = field(default_factory=list)
        vector_split_active: bool = False
        warnings: List[str] = field(default_factory=list)
        reason: str = ""
        confidence: float = 1.0

        def to_dict(self) -> Dict[str, Any]:
            """Convert dataclass to dictionary for backend consumption."""
            from dataclasses import asdict
            return asdict(self)

    @dataclass
    class PipelineResult:
        """Result of the full configuration pipeline execution."""
        success: bool
        suggestion: Optional[ResourceSuggestion] = None
        config_path: Optional[str] = None
        dry_run_content: Optional[str] = None
        validation_errors: List[str] = field(default_factory=list)
        warnings: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Core Imports
# =============================================================================
from .topology import Topology
from .hardware import (
    get_total_mem_kb,
    get_job_memory_limit_mb,
    is_containerized,
    get_scratch_filesystem_type
)
# Import build_auto from builder module to ensure correct single-source-of-truth logic
from .builder import build_auto
from ..optimizer.advisor import suggest_optimal_resources
from ..utils.validation import validate_machines
from ..utils.export import export_config
from ..backend_manager import get_current_backend
from ..logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Helper Wrappers for Hardware & Version Detection
# =============================================================================
def detect_wien2k_version() -> str:
    """
    Detect WIEN2k version via environment or common installation paths.
    Returns a version string for compatibility checks.
    """
    ver = os.getenv("WIEN_VERSION")
    if ver:
        return ver
        
    wienroot = os.getenv("WIENROOT")
    if wienroot:
        version_file = Path(wienroot, "VERSION")
        if version_file.exists():
            try:
                return version_file.read_text().strip().split()[0]
            except Exception:
                pass
    return "2024.x"  # Default fallback

def get_total_ram_gb() -> float:
    """Wrapper to retrieve total system RAM in Gigabytes."""
    return get_total_mem_kb() / (1024.0 * 1024.0)

def get_numa_node_count() -> int:
    """Retrieve the number of NUMA nodes available on the host."""
    try:
        nodes = [d for d in os.listdir("/sys/devices/system/node") if d.startswith("node")]
        return len(nodes)
    except Exception:
        return 1

# =============================================================================
# Preflight Validation Logic
# =============================================================================
def preflight_check(
    topo: Topology,
    suggestion: ResourceSuggestion,
    problem_size: Optional[ProblemSize] = None
) -> List[str]:
    """
    Validate resources against hardware/scheduler limits before execution.
    Enhanced with scratch space, NUMA awareness, and memory heuristics.
    
    Returns:
        List of warnings and errors. Strings starting with "ERROR: " are critical.
    """
    messages = []

    # 1. Memory Estimation Heuristic
    # WIEN2k memory usage scales with N_BANDS * N_KPOINTS and RKMAX.
    # Rough estimate: Base overhead ~ 500MB + (Bands * Kpts * 2KB).
    est_mem_mb = 500.0
    if problem_size:
        band_term = (problem_size.n_bands * problem_size.n_kpoints * 0.002) 
        est_mem_mb += band_term
        if problem_size.has_spin_orbit:
            est_mem_mb *= 1.5  # SOC increases memory footprint significantly
            
    # Check against Job Limit first, then System Limit
    job_limit_mb = get_job_memory_limit_mb()
    total_mem_mb = get_total_ram_gb() * 1024.0
    limit_mb = job_limit_mb if job_limit_mb else total_mem_mb

    # If estimated memory exceeds limit (conservative threshold 90%)
    if est_mem_mb > limit_mb * 0.9:
        messages.append(
            f"ERROR: Estimated memory ({est_mem_mb:.0f}MB) exceeds available limit ({limit_mb:.0f}MB)."
        )
        
    # 2. Scheduler Context Consistency
    if topo.scheduler_hints.get("scheduler") == "slurm" and not os.getenv("SLURM_JOB_ID"):
        messages.append("Warning: SLURM topology detected but SLURM_JOB_ID is missing. Running interactively?")
        
    # 3. NUMA & Binding Awareness
    if suggestion.mode == "mpi" and get_numa_node_count() > 1:
        env_options = os.environ.get("SLURM_OPTIONS", "")
        if "cpu-bind" not in env_options.lower():
            messages.append("Warning: NUMA system detected. Recommend using 'srun --cpu-bind=core' to prevent cross-node traffic.")
            
    # 4. Scratch Space Quality
    scratch_type = get_scratch_filesystem_type()
    if scratch_type in ["nfs", "unknown"]:
        messages.append("Warning: Non-local scratch detected. High I/O latency may degrade LAPW0/LAPW1 performance. Use local SSD/SCRATCH.")
        
    # 5. Containerization Check
    if is_containerized() and topo.scheduler_hints.get("scheduler") != "none":
        messages.append("Warning: Running in container. Ensure container MPI libraries and OpenIB/UCX drivers match the host.")
        
    return messages

# =============================================================================
# Main Pipeline Orchestrator
# =============================================================================
def run_pipeline(
    topo: Topology,
    output_format: str = "json",
    dry_run: bool = False,
    export_path: Optional[str] = None,
    operation_id: Optional[str] = None,
    user_suggestion: Optional[Union[Dict[str, Any], ResourceSuggestion]] = None
) -> PipelineResult:
    """
    Execute the full configuration generation pipeline: Detect -> Advise -> Validate -> Build -> Export.
    
    Args:
        topo: Hardware/Scheduler topology.
        output_format: Format for export (json, yaml, txt).
        dry_run: Generate config string without writing to disk.
        export_path: Path to export the final configuration summary.
        operation_id: Unique ID for logging/tracing.
        user_suggestion: Optional manual override for resources.
    """
    op_id = operation_id or f"pipeline_{int(time.time())}_{os.getpid()}"
    logger.info(f"[{op_id}] Starting WIEN2kGen Pipeline...")

    # Initialize warnings early to prevent UnboundLocalError in except block
    warnings_list: List[str] = []

    try:
        backend = get_current_backend()
        
        # Step 1: Problem Detection
        logger.info(f"[{op_id}] Detecting problem size from backend inputs...")
        try:
            problem = backend.detect_problem_size()
            if isinstance(problem, dict):
                prob_size = ProblemSize(**{k: v for k, v in problem.items() if k in ProblemSize.__dataclass_fields__})
            elif isinstance(problem, ProblemSize):
                prob_size = problem
            else:
                prob_size = ProblemSize()
        except Exception as e:
            logger.warning(f"Problem detection failed: {e}. Using empty ProblemSize.")
            prob_size = ProblemSize()
            
        # Step 2: Resource Suggestion
        logger.info(f"[{op_id}] Generating resource suggestion...")
        if user_suggestion:
            # Normalize user suggestion
            if isinstance(user_suggestion, dict):
                sug_obj = ResourceSuggestion(**user_suggestion)
            elif isinstance(user_suggestion, ResourceSuggestion):
                sug_obj = user_suggestion
            else:
                raise TypeError("Invalid user_suggestion type.")
        else:
            raw_suggestion = suggest_optimal_resources(topo, prob_size)
            if isinstance(raw_suggestion, dict):
                sug_obj = ResourceSuggestion(**raw_suggestion)
            elif isinstance(raw_suggestion, ResourceSuggestion):
                sug_obj = raw_suggestion
            else:
                sug_obj = ResourceSuggestion()
                
        logger.info(f"[{op_id}] Suggestion: {sug_obj.recommended_total_cores} cores, Mode: {sug_obj.mode}")
        
        # Step 3: Preflight Checks
        logger.info(f"[{op_id}] Running preflight checks...")
        checks = preflight_check(topo, sug_obj, prob_size)
        
        warnings_list = [c for c in checks if not c.startswith("ERROR:")]
        errors = [c.replace("ERROR: ", "") for c in checks if c.startswith("ERROR:")]
        
        if warnings_list:
            for w in warnings_list:
                logger.warning(f"[{op_id}] Preflight warning: {w}")
                
        if errors:
            raise ValueError(f"Preflight errors detected: {'; '.join(errors)}")
            
        # Step 4: Build / Dry-Run
        if dry_run:
            logger.info(f"[{op_id}] Dry-run mode: Generating config string...")
            content = backend.generate_input(topo, sug_obj.to_dict())
            logger.info(f"[{op_id}] Dry-run successful. Config preview:")
            for line in content.splitlines()[:5]:
                logger.info(f"  {line}")
            return PipelineResult(
                success=True,
                suggestion=sug_obj,
                dry_run_content=content,
                warnings=warnings_list,
                metadata={"mode": "dry_run"}
            )
            
        # Call the centralized builder
        logger.info(f"[{op_id}] Building configuration files...")
        build_result = build_auto(
            topo=topo,
            suggestion=sug_obj,
            backup=True,
            validate=True
        )
        
        if not build_result.success:
            raise RuntimeError(f"Build failed: {build_result.error_message}")
            
        machines_path = build_result.config_path
        
        # Step 5: Final Validation
        if machines_path and machines_path.exists():
            logger.info(f"[{op_id}] Validating output file: {machines_path}")
            val_result = validate_machines(machines_path)
            if not val_result.get("valid", False):
                raise ValueError(f"Generated configuration invalid: {val_result.get('errors', 'Unknown error')}")
        
        # Step 6: Export
        if export_path:
            logger.info(f"[{op_id}] Exporting configuration summary to {export_path}")
            try:
                export_data = {
                    "topology": topo.to_dict() if hasattr(topo, 'to_dict') else {},
                    "suggestion": sug_obj.to_dict(),
                    "problem_size": prob_size.__dict__ if hasattr(prob_size, '__dict__') else {},
                    "backend": backend.__class__.__name__
                }
                export_config(export_data, Path(export_path), format_hint=output_format)
            except Exception as e:
                logger.warning(f"Export failed: {e}")
                
        logger.info(f"[{op_id}] Pipeline completed successfully")
        return PipelineResult(
            success=True,
            suggestion=sug_obj,
            config_path=str(machines_path) if machines_path else None,
            warnings=warnings_list,
            metadata={"build_result": build_result.to_dict() if hasattr(build_result, 'to_dict') else {}}
        )
        
    except Exception as e:
        logger.error(f"[{op_id}] Pipeline failed: {e}", exc_info=True)
        return PipelineResult(
            success=False,
            validation_errors=[str(e)],
            warnings=warnings_list
        )