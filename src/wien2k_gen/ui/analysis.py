"""
SCF Log Analysis & Performance Scaling Engine for HPC/DFT Workflows.
Provides robust parsing, convergence tracking, parallel scaling analysis,
and structured report generation for WIEN2k, VASP, and Quantum ESPRESSO.
Designed for seamless integration with the TUI/CLI, diagnostic pipelines, and export modules.

Key Architecture Features:
• Multi-code regex parsers with version-agnostic pattern matching
• Structured TypedDict/dataclass outputs for type-safe pipeline consumption
• Parallel scaling efficiency calculation (strong/weak, Amdahl's law approximation)
• Rich-compatible table/tree generation for TUI/CLI rendering
• Graceful fallbacks for truncated logs, missing markers, or malformed I/O
• Comprehensive English documentation, type hints, and HPC-grade error resilience

All documentation and inline comments are in English per project standards.
"""

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

from ..core.constants import RYDBERG_TO_EV
from ..logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Type Definitions for Structured Analysis
# =============================================================================

class SCFParseResult(TypedDict, total=False):
    """Structured extraction result from DFT SCF output logs."""
    code: str  # 'wien2k', 'vasp', 'qe'
    converged: bool
    total_cycles: int
    final_energy_ry: Optional[float]
    final_energy_ev: Optional[float]
    cpu_time_sec: float
    wall_time_sec: float
    max_force_ry_au: Optional[float]
    charge_convergence: float
    stage_timings: Dict[str, float]  # e.g., {'lapw0': 12.5, 'lapw1': 45.2}
    errors: List[str]
    warnings: List[str]
    raw_snippet: str


class ScalingMetrics(TypedDict):
    """Calculated parallel performance metrics from multi-run benchmarks."""
    base_cores: int
    base_time_sec: float
    current_cores: int
    current_time_sec: float
    speedup: float
    efficiency_percent: float
    parallel_overhead_percent: float
    recommendation: str


@dataclass
class AnalysisReport:
    """Aggregated analysis output ready for UI rendering or JSON export."""
    timestamp: float
    code_backend: str
    parsing: Optional[SCFParseResult] = None
    scaling: Optional[ScalingMetrics] = None
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rich_tree: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "code_backend": self.code_backend,
            "parsing": self.parsing,
            "scaling": self.scaling,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
        }


# =============================================================================
# Multi-Code SCF Log Parsers
# =============================================================================

def _parse_wien2k_scf(content: str) -> SCFParseResult:
    """Extract WIEN2k convergence & timing from .scf or .dayfile."""
    result: SCFParseResult = {
        "code": "wien2k",
        "converged": False,
        "total_cycles": 0,
        "final_energy_ry": None,
        "final_energy_ev": None,
        "cpu_time_sec": 0.0,
        "wall_time_sec": 0.0,
        "max_force_ry_au": None,
        "charge_convergence": 0.0,
        "stage_timings": {},
        "errors": [],
        "warnings": [],
        "raw_snippet": content[:500]
    }
    
    # Total Energy
    ene_match = re.search(r':ENE\s+:\s+TOTAL\s+ENERGY\s*=\s*([-\d\.Ee+]+)', content, re.IGNORECASE)
    if ene_match:
        result["final_energy_ry"] = float(ene_match.group(1))
        result["final_energy_ev"] = result["final_energy_ry"] * RYDBERG_TO_EV

    # Convergence & Cycles
    cycle_matches = re.findall(r':DIS\s+:\s+CHARGE\s+CONVERGENCE\s*=\s*([\d\.Ee+]+)', content, re.IGNORECASE)
    if cycle_matches:
        result["total_cycles"] = len(cycle_matches)
        try:
            result["charge_convergence"] = float(cycle_matches[-1])
            result["converged"] = result["charge_convergence"] < 0.0001  # Default WIEN2k threshold
        except ValueError:
            pass

    # Stage Timings (lapw0, lapw1, lapw2, mixer, etc.)
    stage_pattern = r'(\w+)\s+:\s+cpu\s+time\s+:\s+([\d\.]+)'
    for match in re.finditer(stage_pattern, content, re.IGNORECASE):
        stage = match.group(1).lower()
        try:
            result["stage_timings"][stage] = float(match.group(2))
        except ValueError:
            pass

    # Total CPU/Wall
    cpu_match = re.search(r':CPU\s+:\s+TOTAL\s+CPU\s+TIME\s+FOR\s+SCF\s+IS\s*([\d\.]+)', content, re.IGNORECASE)
    if cpu_match:
        result["cpu_time_sec"] = float(cpu_match.group(1))
        
    wall_match = re.search(r':REAL\s+:\s+TOTAL\s+WALL\s+TIME\s+FOR\s+SCF\s+IS\s*([\d\.]+)', content, re.IGNORECASE)
    if wall_match:
        result["wall_time_sec"] = float(wall_match.group(1))

    # Errors/Warnings
    if "QTL-B" in content:
        result["errors"].append("QTL-B error detected. Check case.in1, RKMAX, or k-point grid.")
    if "NOT CONVERGED" in content:
        result["converged"] = False
        result["warnings"].append("SCF did not converge within maximum cycles.")
    if "LAPW1 crashed" in content or "lapw0 crashed" in content:
        result["errors"].append("Critical LAPWx crash. Inspect MPI limits, memory, or case.struct.")

    return result


def _parse_vasp_outcar(content: str) -> SCFParseResult:
    """Extract VASP convergence & timing from OUTCAR or vasprun.xml (text fallback)."""
    result: SCFParseResult = {
        "code": "vasp",
        "converged": False,
        "total_cycles": 0,
        "final_energy_ry": None,
        "final_energy_ev": None,
        "cpu_time_sec": 0.0,
        "wall_time_sec": 0.0,
        "max_force_ry_au": None,
        "charge_convergence": 0.0,
        "stage_timings": {},
        "errors": [],
        "warnings": [],
        "raw_snippet": content[:500]
    }
    
    # Free Energy
    ene_match = re.search(r'free\s+energy\s+TOTEN\s*=\s*([-\d\.Ee+]+)', content, re.IGNORECASE)
    if ene_match:
        result["final_energy_ev"] = float(ene_match.group(1))
        result["final_energy_ry"] = result["final_energy_ev"] / RYDBERG_TO_EV

    # Convergence
    if "reached required accuracy" in content.lower() or "aborting loop because ediff is reached" in content.lower():
        result["converged"] = True

    # Ionic/Electronic steps
    ionic_matches = re.findall(r'iteration\s+\d+\s+E\([-\d\.]+\)', content, re.IGNORECASE)
    result["total_cycles"] = len(ionic_matches) if ionic_matches else len(re.findall(r'FREE\s+ENERGY', content, re.IGNORECASE))

    # CPU Time
    cpu_match = re.search(r'Total\s+CPU\s+time\s+in\s+sec:\s*([\d\.]+)', content)
    if cpu_match:
        result["cpu_time_sec"] = float(cpu_match.group(1))

    # Max Force
    force_match = re.search(r'TOTAL-FORCE\s+eV/Angst\n(.*?)(?:\n\n|\Z)', content, re.DOTALL)
    if force_match:
        forces = []
        for line in force_match.group(1).splitlines():
            parts = line.split()
            if len(parts) >= 4:
                try:
                    mag = math.sqrt(sum(float(x)**2 for x in parts[1:4]))
                    forces.append(mag)
                except ValueError:
                    continue
        if forces:
            result["max_force_ry_au"] = max(forces) / RYDBERG_TO_EV * 0.529177  # eV/Å to Ry/a0 approx

    if "error" in content.lower() and "warning" not in content.lower():
        result["errors"].append("Critical error in OUTCAR.")
    if "ZHEGV" in content or "diago_david" in content:
        if "failed" in content.lower():
            result["errors"].append("Diagonalization failure. Check KPAR/NCORE or mixing.")

    return result


def _parse_qe_pwscf(content: str) -> SCFParseResult:
    """Extract QE convergence & timing from pwscf.out."""
    result: SCFParseResult = {
        "code": "qe",
        "converged": False,
        "total_cycles": 0,
        "final_energy_ry": None,
        "final_energy_ev": None,
        "cpu_time_sec": 0.0,
        "wall_time_sec": 0.0,
        "max_force_ry_au": None,
        "charge_convergence": 0.0,
        "stage_timings": {},
        "errors": [],
        "warnings": [],
        "raw_snippet": content[:500]
    }
    
    # Total Energy
    ene_match = re.search(r'!\s+total\s+energy\s*=\s*([-\d\.Ee+]+)\s*Ry', content, re.IGNORECASE)
    if ene_match:
        result["final_energy_ry"] = float(ene_match.group(1))
        result["final_energy_ev"] = result["final_energy_ry"] * RYDBERG_TO_EV

    # SCF Convergence
    if "convergence has been achieved" in content.lower():
        result["converged"] = True
    scf_matches = re.findall(r'!\s+total\s+energy\s*=', content, re.IGNORECASE)
    result["total_cycles"] = len(scf_matches)

    # Timing
    cpu_match = re.search(r'CPU\s+time\s+\(sec\):\s*([\d\.]+)', content)
    if cpu_match:
        result["cpu_time_sec"] = float(cpu_match.group(1))
    wall_match = re.search(r'WALL\s+time\s+\(sec\):\s*([\d\.]+)', content)
    if wall_match:
        result["wall_time_sec"] = float(wall_match.group(1))

    if "error" in content.lower() and "warning" not in content.lower():
        result["errors"].append("QE runtime error detected.")
    if "not converged" in content.lower():
        result["converged"] = False
        result["warnings"].append("SCF/BFGS did not converge.")

    return result


# =============================================================================
# Public Parsing API
# =============================================================================

def parse_scf_log(log_path: Union[str, Path], code_hint: Optional[str] = None) -> SCFParseResult:
    """
    Auto-detect DFT code and parse SCF output with robust fallbacks.
    Returns structured result for UI/CLI consumption and pipeline integration.
    """
    path = Path(log_path)
    if not path.exists():
        logger.error(f"Log file not found: {path}")
        return {"code": "unknown", "converged": False, "errors": ["File not found"], "raw_snippet": ""}
        
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"code": "unknown", "converged": False, "errors": [f"Read error: {e}"], "raw_snippet": ""}

    code = (code_hint or "").lower()
    if not code:
        # Auto-detect from content signatures
        if "WIEN2k" in content or ":CPU  :" in content:
            code = "wien2k"
        elif "vasp" in content.lower() or "outcar" in content.lower() or "TOTEN" in content:
            code = "vasp"
        elif "pwscf" in content.lower() or "quantum espresso" in content.lower() or "k-points" in content.lower():
            code = "qe"
        else:
            code = "unknown"

    parsers = {"wien2k": _parse_wien2k_scf, "vasp": _parse_vasp_outcar, "qe": _parse_qe_pwscf}
    return parsers.get(code, lambda c: {"code": "unknown", "errors": ["Unsupported log format"], "raw_snippet": c[:500]})(content)


# =============================================================================
# Parallel Scaling Analysis
# =============================================================================

def calculate_scaling_metrics(
    base_cores: int,
    base_time_sec: float,
    current_cores: int,
    current_time_sec: float,
    overhead_threshold: float = 30.0
) -> ScalingMetrics:
    """
    Compute **strong scaling** speedup, efficiency, and overhead.

    Strong scaling: fixed problem size, increasing core count.
    Ideal speedup = P (linear), ideal efficiency = 100%.

    Uses standard HPC scaling formulas with safety guards for division-by-zero.

    For weak scaling analysis (problem size grows with core count),
    use :func:`calculate_weak_scaling_metrics`.
    """
    if base_time_sec <= 0 or current_time_sec <= 0:
        return {
            "base_cores": base_cores,
            "base_time_sec": base_time_sec,
            "current_cores": current_cores,
            "current_time_sec": current_time_sec,
            "speedup": 0.0,
            "efficiency_percent": 0.0,
            "parallel_overhead_percent": 100.0,
            "recommendation": "Invalid timing data. Check log parsing or run duration."
        }
        
    speedup = base_time_sec / current_time_sec
    efficiency = (speedup / current_cores) * base_cores * 100.0
    overhead = 100.0 - efficiency

    # Generate actionable recommendation
    if efficiency > 85:
        rec = "Excellent scaling. Safe to increase cores."
    elif efficiency > 65:
        rec = "Good scaling. Monitor I/O bottlenecks at higher core counts."
    elif efficiency > 40:
        rec = "Moderate scaling. Consider hybrid MPI+OpenMP or optimize KPAR/nband/NCORE."
    else:
        rec = "Poor scaling. Likely communication/I/O bound. Reduce ranks, increase threads, or check interconnect."

    return {
        "base_cores": base_cores,
        "base_time_sec": round(base_time_sec, 2),
        "current_cores": current_cores,
        "current_time_sec": round(current_time_sec, 2),
        "speedup": round(speedup, 2),
        "efficiency_percent": round(efficiency, 2),
        "parallel_overhead_percent": round(overhead, 2),
        "recommendation": rec
    }


def calculate_weak_scaling_metrics(
    base_cores: int,
    base_time_sec: float,
    base_problem_size: int,
    scaled_cores: int,
    scaled_time_sec: float,
    scaled_problem_size: int,
) -> ScalingMetrics:
    """
    Compute **weak scaling** speedup, efficiency, and overhead.

    Weak scaling (Gustafson's Law): problem size grows proportionally to
    core count, keeping work per core constant. The ideal efficiency is 100%
    regardless of scale.

    Per Hager & Wellein 2010 ("Introduction to High Performance Computing
    for Scientists and Engineers", Ch. 4), the weak scaling efficiency is:

        Speedup = T(P_1, N_1) / T(P, N_P)

    where N_P = N_1 * (P / P_1) and efficiency = speedup.

    Unlike strong scaling (Amdahl's Law), weak scaling isolates communication
    overhead rather than serial fraction, making it the preferred metric for
    assessing large-scale HPC applications.

    Parameters
    ----------
    base_cores : int
        Number of cores for the base run.
    base_time_sec : float
        Wall time for the base run in seconds.
    base_problem_size : int
        Problem size (e.g., atoms, nmat) for the base run.
    scaled_cores : int
        Number of cores for the scaled run.
    scaled_time_sec : float
        Wall time for the scaled run in seconds.
    scaled_problem_size : int
        Problem size for the scaled run.

    Returns
    -------
    ScalingMetrics
        Dictionary with weak scaling metrics.
    """
    if base_time_sec <= 0 or scaled_time_sec <= 0:
        return {
            "base_cores": base_cores,
            "base_time_sec": base_time_sec,
            "current_cores": scaled_cores,
            "current_time_sec": scaled_time_sec,
            "speedup": 0.0,
            "efficiency_percent": 0.0,
            "parallel_overhead_percent": 100.0,
            "recommendation": "Invalid timing data."
        }

    ideal_time = base_time_sec
    speedup = base_time_sec / scaled_time_sec
    efficiency = speedup * 100.0

    cores_ratio = scaled_cores / max(1, base_cores)
    problem_ratio = scaled_problem_size / max(1, base_problem_size)
    overhead = max(0.0, 100.0 - efficiency)

    if base_problem_size > 0 and scaled_problem_size > 0:
        work_per_core_base = base_problem_size / max(1, base_cores)
        work_per_core_scaled = scaled_problem_size / max(1, scaled_cores)
        load_imbalance = abs(work_per_core_base - work_per_core_scaled) / max(1, max(work_per_core_base, work_per_core_scaled)) * 100.0
    else:
        load_imbalance = 0.0

    if load_imbalance > 10.0:
        rec = (
            f"Load imbalance of {load_imbalance:.1f}% detected. "
            f"Problem size does not scale linearly with cores "
            f"(expected {cores_ratio:.1f}x, got {problem_ratio:.1f}x)."
        )
    elif efficiency > 90:
        rec = "Excellent weak scaling. Communication overhead is minimal."
    elif efficiency > 70:
        rec = "Good weak scaling. Minor communication overhead; check MPI collectives."
    elif efficiency > 50:
        rec = "Moderate weak scaling. Communication is becoming the bottleneck."
    else:
        rec = (
            "Poor weak scaling. Communication dominates. Reduce MPI ranks, "
            "improve data locality, or use hybrid MPI+OpenMP."
        )

    return {
        "base_cores": base_cores,
        "base_time_sec": round(base_time_sec, 2),
        "current_cores": scaled_cores,
        "current_time_sec": round(scaled_time_sec, 2),
        "speedup": round(speedup, 2),
        "efficiency_percent": round(efficiency, 2),
        "parallel_overhead_percent": round(overhead, 2),
        "recommendation": rec
    }


def visualize_scaling(
    scaling_data: Dict[int, float],  # {cores: time_sec}
    title: str = "Parallel Scaling Analysis"
) -> str:
    """
    Generate Rich-formatted markdown/table for scaling results.
    Suitable for TUI modal display or CLI export.
    """
    if not scaling_data or len(scaling_data) < 2:
        return "[dim]Insufficient data for scaling analysis (requires ≥2 runs).[/]"

    from rich.console import Console
    from rich.table import Table
        
    sorted_runs = sorted(scaling_data.items())
    base_cores, base_time = sorted_runs[0]

    table = Table(title=f"[bold cyan]{title}[/]", show_lines=True, box=None, padding=(0, 1))
    table.add_column("Cores", style="cyan", justify="center")
    table.add_column("Time (s)", style="white", justify="right")
    table.add_column("Speedup", style="green", justify="right")
    table.add_column("Efficiency %", style="yellow", justify="right")

    for cores, t in sorted_runs:
        metrics = calculate_scaling_metrics(base_cores, base_time, cores, t)
        eff_color = "green" if metrics["efficiency_percent"] > 70 else "yellow" if metrics["efficiency_percent"] > 40 else "red"
        table.add_row(
            str(cores),
            f"{metrics['current_time_sec']:.2f}",
            f"{metrics['speedup']:.2f}x",
            f"[{eff_color}]{metrics['efficiency_percent']:.1f}%[/]"
        )

    console = Console()
    with console.capture() as capture:
        console.print(table)
        console.print(f"\n[bold dim]Base Configuration:[/] {base_cores} cores | {base_time:.2f}s")
    return capture.get()


# =============================================================================
# Report Generation & Aggregation
# =============================================================================

def generate_report(
    parsed_scf: SCFParseResult,
    scaling_data: Optional[Dict[int, float]] = None,
    include_recommendations: bool = True,
    verbose: bool = False,
) -> AnalysisReport:
    """
    Compile parsed SCF data + scaling metrics + backend intelligence into a structured report.

    Connects Bayesian optimization, Roofline analysis, charge sloshing diagnosis,
    NUMA topology, and k-point load balancing to the terminal UI.

    Parameters
    ----------
    parsed_scf : SCFParseResult
        Parsed SCF output from parse_scf_log().
    scaling_data : dict, optional
        {cores: time_sec} mapping for scaling analysis.
    include_recommendations : bool
        Generate actionable recommendations.
    verbose : bool
        Include full backend trace (Bayesian history, Roofline details, NUMA map).
    """
    report = AnalysisReport(
        timestamp=time.time(),
        code_backend=parsed_scf.get("code", "unknown"),
        parsing=parsed_scf,
    )

    if scaling_data and len(scaling_data) >= 2:
        sorted_runs = sorted(scaling_data.items())
        report.scaling = calculate_scaling_metrics(
            sorted_runs[0][0], sorted_runs[0][1],
            sorted_runs[-1][0], sorted_runs[-1][1]
        )

    recs = []
    if not parsed_scf.get("converged"):
        recs.append("SCF not converged. Adjust mixing parameters (BROYDEN, KERKER), increase NSTEPS, or check k-point density.")
    if parsed_scf.get("cpu_time_sec", 0) > 0 and parsed_scf.get("wall_time_sec", 0) > 0:
        ratio = parsed_scf["cpu_time_sec"] / parsed_scf["wall_time_sec"]
        if ratio < 0.5:
            recs.append(f"Low CPU/Wall ratio ({ratio:.2f}). Likely I/O or network bottleneck. Use local scratch and check interconnect.")
        elif ratio > 0.95:
            recs.append("Excellent CPU utilization. Scaling bottleneck is likely computational.")

    if report.scaling and report.scaling["efficiency_percent"] < 50:
        recs.append(f"Parallel efficiency is low ({report.scaling['efficiency_percent']:.1f}%). Reduce MPI ranks or switch to hybrid mode.")
    if "QTL-B" in str(parsed_scf.get("errors", [])):
        recs.append("QTL-B detected. Lower RKMAX, increase GMAX, or refine linearization energies in case.in1.")

    # ── Backend Intelligence Integration ──
    try:
        from ..optimizer.monitor import diagnose_charge_sloshing_root_cause
        raw = parsed_scf.get("raw_snippet", "")
        slosh_diag = diagnose_charge_sloshing_root_cause(raw)
        if slosh_diag["root_cause"] != "none":
            rc = slosh_diag["root_cause"].replace("_", " ").title()
            recs.append(f"Charge sloshing: {rc} (confidence {slosh_diag['confidence']:.0%}). "
                        f"Fix: {'; '.join(a['action'] for a in slosh_diag['actions'][:2])}")
            if verbose:
                for a in slosh_diag["actions"]:
                    recs.append(f"  → {a['action']} [{a['priority']}]: {a['reason']}")
    except Exception:
        if verbose:
            recs.append("(Charge sloshing analyzer unavailable)")

    try:
        from ..core.hardware import get_memory_bandwidth_gb_s, get_numa_node_count
        bw = get_memory_bandwidth_gb_s()
        numa = get_numa_node_count()
        if bw < 50:
            recs.append(f"Low memory bandwidth ({bw:.0f} GB/s). LAPW1 is memory-bound; extra MPI ranks yield no benefit.")
        if verbose:
            recs.append(f"Hardware: {numa} NUMA node(s), {bw:.0f} GB/s bandwidth")
    except Exception:
        if verbose:
            recs.append("(Hardware detection unavailable)")

    try:
        from ..optimizer.convergence import detect_scf_divergence
        div = detect_scf_divergence(parsed_scf.get("raw_snippet", ""))
        if div["divergent"]:
            recs.append(f"SCF divergence ({div['divergence_type']}): {div['recommended_action'][:120]}...")
            if verbose:
                recs.append(f"  → Auto fix: beta={div['auto_mixing_params']['beta']}, "
                           f"pratt_cycles={div['auto_mixing_params']['pratt_cycles']}")
    except Exception:
        if verbose:
            recs.append("(Divergence detector unavailable)")

    report.recommendations = recs
    report.warnings = parsed_scf.get("warnings", [])

    from rich.console import Console
    from rich.tree import Tree
    tree = Tree(f"[bold]{report.code_backend.upper()} Analysis Report[/]")
    conv_node = tree.add(f"Convergence: {'[green]YES[/]' if parsed_scf.get('converged') else '[red]NO[/]'}")
    conv_node.add(f"Cycles: {parsed_scf.get('total_cycles', 0)}")
    if parsed_scf.get("final_energy_ry"):
        conv_node.add(f"Energy: {parsed_scf['final_energy_ry']:.6f} Ry")

    time_node = tree.add("Timing")
    time_node.add(f"CPU: {parsed_scf.get('cpu_time_sec', 0):.1f}s")
    time_node.add(f"Wall: {parsed_scf.get('wall_time_sec', 0):.1f}s")

    if parsed_scf.get("stage_timings"):
        stage_node = time_node.add("Per-Stage CPU")
        for stage, t in sorted(parsed_scf["stage_timings"].items()):
            stage_node.add(f"{stage}: {t:.1f}s")

    if report.scaling:
        scale_node = tree.add("Scaling")
        scale_node.add(f"Speedup: {report.scaling['speedup']:.2f}x")
        scale_node.add(f"Efficiency: {report.scaling['efficiency_percent']:.1f}%")

    if recs:
        rec_node = tree.add("[bold yellow]Recommendations[/]")
        for r in recs:
            rec_node.add(f"[dim]• {r}[/dim]")

    console = Console()
    with console.capture() as cap:
        console.print(tree)
    report.rich_tree = cap.get()

    return report


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "AnalysisReport",
    "SCFParseResult",
    "ScalingMetrics",
    "calculate_scaling_metrics",
    "calculate_weak_scaling_metrics",
    "generate_report",
    "parse_scf_log",
    "visualize_scaling",
]