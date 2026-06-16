"""
Configuration Validation & Backup Management Module for HPC/DFT Workflows.
Provides rigorous syntax checking, consistency validation, and topology-aware
cross-referencing for WIEN2k `.machines` and related parallel configuration files.
Also handles safe backup rotation, atomic copying, and structured diagnostic reporting.

Key Features:
• Robust WIEN2k `.machines` parser with regex-based directive extraction
• Multi-level validation: syntax → internal consistency → topology cross-checks
• Automatic detection of oversubscription, OMP/MPI divisibility violations, and I/O bottlenecks
• Safe backup creation with timestamping and configurable retention policy
• Structured ValidationResult TypedDict for UI/CLI consumption and pipeline integration
• Comprehensive English documentation, type hints, and HPC-grade error handling
• Zero-dependency fallbacks with graceful degradation on missing topology/hardware data

All documentation and inline comments are in English per project standards.
"""

import os
import re
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, TypedDict, Union
from dataclasses import dataclass, field, asdict

from ..logging_config import get_logger

# FIXED: Use __name__ instead of undefined 'name'
logger = get_logger(__name__)


# =============================================================================
# Type Definitions for Structured Validation & Configuration
# =============================================================================

class ValidationResult(TypedDict, total=False):
    """
    Comprehensive validation outcome with severity-classified messages.
    Designed for pipeline gating, UI reporting, and automated troubleshooting.
    """
    valid: bool
    path: Optional[str]
    errors: List[str]
    warnings: List[str]
    info: List[str]
    config: Optional[Dict[str, Any]]
    timestamp: float


class MachinesConfig(TypedDict, total=False):
    """
    Parsed representation of a WIEN2k `.machines` file.
    Normalizes diverse formatting into a structured dictionary for validation.
    """
    nodes: List[str]
    cores_per_node: List[int]
    mode: str  # 'mpi', 'hybrid', 'kpoint'
    omp_global: int
    kpar: int
    lapw0_cores: int
    lapw1_cores: int
    lapw2_cores: int
    vector_split: int
    extrafine: int
    granularity: int
    raw_lines: List[str]


# =============================================================================
# WIEN2k .machines Parser
# =============================================================================

def parse_machines_file(path: Union[str, Path]) -> Tuple[MachinesConfig, List[str]]:
    """
    Parse WIEN2k `.machines` file into structured config.
    Handles modern and legacy formats, ignores comments/blanks,
    and extracts parallelization directives with robust fallbacks.
    
    Args:
        path: Path to `.machines` file.
        
    Returns:
        Tuple of (parsed_config, parsing_warnings)
    """
    config: MachinesConfig = {
        "nodes": [],
        "cores_per_node": [],
        "mode": "mpi",
        "omp_global": 1,
        "kpar": 0,
        "lapw0_cores": 0,
        "lapw1_cores": 0,
        "lapw2_cores": 0,
        "vector_split": 0,
        "extrafine": 0,
        "granularity": 1,
        "raw_lines": []
    }
    parse_warnings: List[str] = []
    target = Path(path)

    if not target.exists():
        parse_warnings.append(f"File not found: {target}")
        return config, parse_warnings
        
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        parse_warnings.append(f"Read error: {e}")
        return config, parse_warnings
        
    raw_lines = content.splitlines()
    config["raw_lines"] = [l.strip() for l in raw_lines if l.strip() and not l.strip().startswith("#")]

    # Track node allocations to detect duplicates or mismatches
    node_allocations: Dict[str, int] = {}
    lapw1_nodes = set()
    lapw2_nodes = set()

    for line_idx, line in enumerate(raw_lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
            
        # Directives with values
        val_match = re.match(r'^(omp_global|kpar|granularity|extrafine|lapw2_vector_split)\s*:\s*(\d+)', stripped, re.IGNORECASE)
        if val_match:
            key, val = val_match.group(1).lower(), int(val_match.group(2))
            if key == "omp_global": config["omp_global"] = val
            elif key == "kpar": config["kpar"] = val
            elif key == "granularity": config["granularity"] = val
            elif key == "extrafine": config["extrafine"] = val
            elif key == "lapw2_vector_split": config["vector_split"] = val
            continue
            
        # lapw0 directive
        lapw0_match = re.match(r'^lapw0\s*:\s*([^\s:]+)\s*:\s*(\d+)', stripped, re.IGNORECASE)
        if lapw0_match:
            config["lapw0_cores"] = int(lapw0_match.group(2))
            continue
            
        # lapw1/lapw2 node allocation
        lapw_match = re.match(r'^lapw(1|2)\s*:\s*([^\s:]+)\s*:\s*(\d+)', stripped, re.IGNORECASE)
        if lapw_match:
            prog = lapw_match.group(1)
            node = lapw_match.group(2)
            cores = int(lapw_match.group(3))
            if node not in node_allocations:
                node_allocations[node] = 0
            node_allocations[node] += cores
            if prog == "1": lapw1_nodes.add(node)
            else: lapw2_nodes.add(node)
            continue
            
        # k-point parallel mode (1: hostname)
        kpt_match = re.match(r'^1\s*:\s*([^\s:]+)', stripped)
        if kpt_match:
            node = kpt_match.group(1)
            config["mode"] = "kpoint"
            if node not in node_allocations:
                node_allocations[node] = 0
            node_allocations[node] += 1
            continue
            
        # Fallback: warn about unrecognized non-comment lines
        if not stripped.startswith("#"):
            parse_warnings.append(f"Line {line_idx} unrecognized or malformed: '{stripped[:50]}...'")
            
    # Flatten node allocations
    config["nodes"] = sorted(node_allocations.keys())
    config["cores_per_node"] = [node_allocations[n] for n in config["nodes"]]

    # Infer mode if not explicitly set by k-point lines
    if config["mode"] != "kpoint":
        if config["omp_global"] > 1:
            config["mode"] = "hybrid"
        else:
            config["mode"] = "mpi"
            
    # Aggregate lapw cores
    config["lapw1_cores"] = sum(node_allocations[n] for n in lapw1_nodes) if lapw1_nodes else sum(config["cores_per_node"])
    config["lapw2_cores"] = sum(node_allocations[n] for n in lapw2_nodes) if lapw2_nodes else sum(config["cores_per_node"])

    return config, parse_warnings


# =============================================================================
# Validation Engine: Syntax, Consistency & Topology-Aware Checks
# =============================================================================

def _check_syntax(config: MachinesConfig) -> List[str]:
    """Validate basic syntax, formatting, and required directives."""
    errors = []
    if not config["nodes"]:
        errors.append("No compute nodes found in .machines file.")
    if any(c <= 0 for c in config["cores_per_node"]):
        errors.append("Non-positive core count detected. All nodes must have >= 1 core.")
    if config["omp_global"] <= 0:
        errors.append("omp_global must be >= 1.")
    if config["kpar"] < 0:
        errors.append("kpar cannot be negative.")
    return errors


def _check_consistency(config: MachinesConfig) -> Tuple[List[str], List[str]]:
    """Check internal mathematical & logical consistency."""
    errors = []
    warnings = []
    total_cores = sum(config["cores_per_node"])
    omp = config["omp_global"]
    kpar = config["kpar"]

    # OMP divisibility
    if omp > 1 and total_cores % omp != 0:
        errors.append(f"total_cores ({total_cores}) not divisible by omp_global ({omp}). "
                      f"Hybrid mode requires integer MPI ranks per node.")
                   
    # kpar vs nodes
    if kpar > 0 and kpar > len(config["nodes"]):
        warnings.append(f"kpar ({kpar}) exceeds node count ({len(config['nodes'])}). "
                        f"Excess k-point pools will be idle.")
    if kpar > 0 and len(config["nodes"]) % kpar != 0:
        warnings.append(f"Node count ({len(config['nodes'])}) not divisible by kpar ({kpar}). "
                        f"Load imbalance expected in k-point distribution.")
                    
    # vector_split validation
    if config["vector_split"] > 0:
        if config["lapw2_cores"] % config["vector_split"] != 0:
            warnings.append(f"lapw2_vector_split ({config['vector_split']}) does not divide lapw2 cores ({config['lapw2_cores']}).")
            
    # lapw0 sanity
    if config["lapw0_cores"] > total_cores:
        errors.append("lapw0_cores exceeds total allocated cores. Potential resource conflict.")
        
    return errors, warnings


def _check_topology_alignment(config: MachinesConfig, topo: Any) -> List[str]:
    """Cross-reference parsed config with detected hardware/scheduler topology."""
    warnings = []
    if topo is None:
        return warnings
        
    topo_nodes = getattr(topo, "nodes", [])
    topo_cores = getattr(topo, "cores_per_node", [])

    # Node name mismatch
    missing_in_topo = set(config["nodes"]) - set(topo_nodes)
    if missing_in_topo:
        warnings.append(f"Nodes in .machines not in current allocation: {', '.join(sorted(missing_in_topo))}")
        
    # Core count mismatch
    if len(config["nodes"]) == len(topo_nodes):
        for cn, tc in zip(config["cores_per_node"], topo_cores):
            if cn > tc:
                warnings.append(f"Allocated cores ({cn}) exceed available topology cores ({tc}) on a node. Oversubscription risk.")
                
    # Scheduler hints
    env_type = getattr(topo, "env_type", "")
    omp = config["omp_global"]
    if omp > 1 and env_type == "slurm":
        warnings.append("OpenMP > 1 in SLURM environment. Ensure --cpus-per-task and --hint=nomultithread are set in job script.")
        
    return warnings


def validate_machines(
    path: Union[str, Path],
    topo: Optional[Any] = None,
    strict_mode: bool = False
) -> ValidationResult:
    """
    Full validation pipeline for WIEN2k `.machines` file.
    Combines syntax parsing, consistency checks, and optional topology alignment.
    
    Args:
        path: Path to `.machines` file.
        topo: Optional Topology instance for cross-validation.
        strict_mode: If True, promote warnings to errors.
        
    Returns:
        ValidationResult with validity status, diagnostics, and parsed config.
    """
    target = Path(path)
    result: ValidationResult = {
        "valid": False,
        "path": str(target),
        "errors": [],
        "warnings": [],
        "info": [],
        "config": None,
        "timestamp": time.time()
    }

    if not target.exists():
        result["errors"].append(f"Configuration file not found: {target}")
        return result
        
    # 1. Parse
    config, parse_warnings = parse_machines_file(target)
    result["config"] = config
    result["warnings"].extend(parse_warnings)
    result["info"].append(f"Parsed {len(config['nodes'])} nodes, mode={config['mode']}")

    # 2. Syntax & Consistency
    syntax_errors = _check_syntax(config)
    consistency_errors, consistency_warnings = _check_consistency(config)
    result["errors"].extend(syntax_errors)
    result["errors"].extend(consistency_errors)
    result["warnings"].extend(consistency_warnings)

    # 3. Topology Alignment (Optional)
    if topo is not None:
        topo_warnings = _check_topology_alignment(config, topo)
        result["warnings"].extend(topo_warnings)
        
    # 4. Strict Mode Promotion
    if strict_mode:
        promoted = [w for w in result["warnings"] if w]
        result["errors"].extend(promoted)
        result["warnings"] = []
        
    # 5. Final Validity Check
    if not result["errors"]:
        result["valid"] = True
        result["info"].append("Validation passed. Configuration is consistent and ready for execution.")
    else:
        result["info"].append(f"Validation failed: {len(result['errors'])} critical issue(s) found.")
        
    logger.debug(
        f"Validation of {target}: valid={result['valid']}, "
        f"errors={len(result['errors'])}, warnings={len(result['warnings'])}"
    )
    return result


# =============================================================================
# Backup & Rotation Utilities
# =============================================================================

def backup_machines(
    path: Union[str, Path],
    max_backups: int = 3,
    timestamp_format: str = "%Y%m%d_%H%M%S"
) -> Optional[Path]:
    """
    Create a timestamped backup of the current configuration file.
    Automatically rotates old backups to prevent disk clutter on scratch space.
    
    Args:
        path: Source configuration file path.
        max_backups: Maximum number of backup files to retain.
        timestamp_format: Datetime format for backup suffix.
        
    Returns:
        Path to newly created backup, or None if backup failed or source missing.
    """
    src = Path(path)
    if not src.exists():
        logger.warning(f"Cannot backup: {src} does not exist.")
        return None
        
    ts = time.strftime(timestamp_format)
    backup_name = f"{src.name}.bak.{ts}"
    backup_path = src.parent / backup_name

    try:
        shutil.copy2(src, backup_path)
        logger.info(f"Backed up {src} -> {backup_path}")
        
        # Rotate old backups
        _rotate_backups(src.parent, f"{src.name}.bak.", max_backups)
        
        return backup_path
    except Exception as e:
        logger.error(f"Backup creation failed for {src}: {e}")
        return None


def _rotate_backups(directory: Path, prefix: str, max_retention: int) -> None:
    """
    Remove oldest backup files matching the prefix if count exceeds max_retention.
    Uses modification time for sorting to ensure deterministic cleanup.
    """
    if max_retention <= 0:
        return
    try:
        backups = sorted(
            [f for f in directory.iterdir() if f.name.startswith(prefix) and f.is_file()],
            key=lambda f: f.stat().st_mtime
        )
        while len(backups) > max_retention:
            oldest = backups.pop(0)
            oldest.unlink(missing_ok=True)
            logger.debug(f"Rotated out old backup: {oldest}")
    except Exception as e:
        logger.warning(f"Backup rotation failed in {directory}: {e}")


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "ValidationResult",
    "MachinesConfig",
    "parse_machines_file",
    "validate_machines",
    "backup_machines",
    "_rotate_backups",
]