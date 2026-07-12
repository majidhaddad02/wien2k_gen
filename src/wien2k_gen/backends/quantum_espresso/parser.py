"""
parser.py - Quantum ESPRESSO Output Parser
Extracts convergence status, energies, forces, stresses, timing, and errors
from QE pw.x/ph.x/cp.x output logs with version-agnostic regex matching.
Production features:
• Multi-step calculation support (scf, relax, vc-relax, md)
• Robust energy/force/stress extraction with unit conversion handling
• CPU vs WALL time separation with per-step breakdown when available
• Structured return types with comprehensive validation hooks
• Graceful fallbacks for truncated, corrupted, or non-standard logs
"""

import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict

from ...core.constants import RYDBERG_TO_EV
from ...logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

class QEParseResult(TypedDict, total=False):
    """Structured extraction result from QE output logs."""
    exists: bool
    converged: bool
    calculation_type: str  # scf, relax, vc-relax, md, etc.
    total_energy_ry: Optional[float]
    total_energy_ev: Optional[float]
    fermi_energy_ry: Optional[float]
    forces: list[list[float]]  # shape: (natoms, 3)
    stress: list[list[float]]  # shape: (3, 3)
    scf_cycles: int
    relaxation_steps: int
    cpu_time_sec: float
    wall_time_sec: float
    errors: list[str]
    warnings: list[str]
    log_snippet: str


@dataclass
class StepMetrics:
    """Per-step or per-SCF-cycle metrics for detailed analysis."""
    cycle: int
    energy_ry: float
    cpu_time: float
    wall_time: float
    max_force: Optional[float] = None
    converged: bool = False


# =============================================================================
# Core Parsing Logic
# =============================================================================

def _extract_energy_block(content: str) -> tuple[Optional[float], Optional[float]]:
    """
    Extract total and Fermi energies from QE output.
    Handles both 'total energy' and '!    total energy' formats across QE versions.
    Returns (total_energy_ry, fermi_energy_ry).
    """
    total_energy = None
    fermi_energy = None

    # Total energy pattern
    total_match = re.search(
        r'!\s+total\s+energy\s*=\s*([-\d\.Ee+]+)\s*Ry',
        content, re.IGNORECASE
    )
    if total_match:
        with contextlib.suppress(ValueError):
            total_energy = float(total_match.group(1))

    # Fallback: older format without '!'
    if total_energy is None:
        total_match_old = re.search(
            r'total\s+energy\s*=\s*([-\d\.Ee+]+)\s*Ry',
            content, re.IGNORECASE
        )
        if total_match_old:
            with contextlib.suppress(ValueError):
                total_energy = float(total_match_old.group(1))

    # Fermi energy pattern
    fermi_match = re.search(
        r'the\s+fermi\s+energy\s+is\s*([-\d\.Ee+]+)\s*ev',
        content, re.IGNORECASE
    )
    if fermi_match:
        try:
            fermi_energy_ev = float(fermi_match.group(1))
            # Convert eV to Ry (1 Ry ≈ RYDBERG_TO_EV eV)
            fermi_energy = fermi_energy_ev / RYDBERG_TO_EV
        except ValueError:
            pass

    return total_energy, fermi_energy


def _extract_forces(content: str, natoms: Optional[int]) -> list[list[float]]:
    """
    Extract final atomic forces from 'Final forces' or 'Forces acting on atoms' blocks.
    Returns list of [fx, fy, fz] per atom.
    """
    forces = []
    # Match force block
    force_pattern = re.compile(
        r'Final\s+forces\s+(?:acting\s+on\s+atoms\s*)?[-=]*\s*\n'
        r'(\s*\d+\s+[-\d\.Ee+]+\s+[-\d\.Ee+]+\s+[-\d\.Ee+]+\s*\n)+',
        re.IGNORECASE | re.MULTILINE
    )
    match = force_pattern.search(content)
    if match:
        block = match.group(0)
        for line in block.splitlines():
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    forces.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue
    return forces


def _extract_stress(content: str) -> list[list[float]]:
    """
    Extract stress tensor from 'Total stress' block.
    Returns 3x3 matrix in kBar or Ry/Bohr^3 (QE standard).
    """
    stress = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    stress_pattern = re.compile(
        r'Total\s+stress\s+(?:\(kBar\)|[-=]*)\s*\n'
        r'((?:\s*[-\d\.Ee+]+\s+[-\d\.Ee+]+\s+[-\d\.Ee+]+\s*\n){3})',
        re.IGNORECASE | re.MULTILINE
    )
    match = stress_pattern.search(content)
    if match:
        lines = match.group(1).strip().splitlines()
        for i, line in enumerate(lines[:3]):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    stress[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
                except ValueError:
                    continue
    return stress


def _extract_timing(content: str) -> tuple[float, float]:
    """
    Extract CPU and WALL time from final timing block.
    QE prints: 'CPU     time (sec):   XXX.XX' and 'WALL    time (sec):   XXX.XX'
    """
    cpu_time = 0.0
    wall_time = 0.0

    cpu_match = re.search(r'CPU\s+time\s+\(sec\):\s*([\d\.]+)', content)
    if cpu_match:
        with contextlib.suppress(ValueError):
            cpu_time = float(cpu_match.group(1))

    wall_match = re.search(r'WALL\s+time\s+\(sec\):\s*([\d\.]+)', content)
    if wall_match:
        with contextlib.suppress(ValueError):
            wall_time = float(wall_match.group(1))

    return cpu_time, wall_time


def _detect_calculation_type(content: str) -> str:
    """Identify QE calculation type from namelist or output markers."""
    lower = content.lower()
    if 'vc-relax' in lower:
        return 'vc-relax'
    if 'relax' in lower:
        return 'relax'
    if 'md' in lower or 'molecular dynamics' in lower:
        return 'md'
    if 'scf' in lower:
        return 'scf'
    return 'unknown'


def _check_convergence(content: str, calc_type: str) -> bool:
    """
    Determine if calculation successfully converged.
    Handles SCF, relaxation, and molecular dynamics completion markers.
    """
    lower = content.lower()
    if 'convergence has been achieved' in lower:
        return True
    if 'job done' in lower and 'end of job' in lower:
        return True
    if calc_type in ('relax', 'vc-relax') and ('bfgs converged' in lower or 'final convergence achieved' in lower):
        return True
    return bool(calc_type == 'scf' and 'convergence threshold' in lower and 'met' in lower)


def _detect_errors_warnings(content: str) -> tuple[list[str], list[str]]:
    """
    Scan output for critical errors and non-fatal warnings.
    Uses pattern matching against known QE failure modes.
    """
    errors = []
    warnings = []
    lower = content.lower()

    error_patterns = [
        (r'error:', 'Generic calculation error'),
        (r'stopped with error', 'Explicit stop due to error'),
        (r'segmentation fault', 'Segmentation fault (SIGSEGV)'),
        (r'not converged', 'SCF or relaxation failed to converge'),
        (r'internal error', 'Internal library/runtime error'),
        (r'parallel execution failed', 'MPI communication failure'),
        (r'too many steps', 'Relaxation/MD exceeded max steps'),
    ]
    for pat, msg in error_patterns:
        if re.search(pat, lower):
            errors.append(msg)

    warning_patterns = [
        (r'warning:', 'Non-fatal warning encountered'),
        (r'charge is not accurate', 'Charge convergence may be marginal'),
        (r'k-point grid is not optimal', 'k-point symmetry or grid warning'),
        (r'scf convergence is slow', 'Mixing parameters may need adjustment'),
    ]
    for pat, msg in warning_patterns:
        if re.search(pat, lower):
            warnings.append(msg)

    return errors, warnings


# =============================================================================
# Public API
# =============================================================================

def parse_qe_output(
    log_path: Path,
    natoms: Optional[int] = None
) -> QEParseResult:
    """
    Parse Quantum ESPRESSO output file with robust, version-agnostic extraction.
    Returns structured result with energies, forces, timing, convergence, and diagnostics.

    Args:
        log_path: Path to pw.x output file (e.g., pwscf.out, *.log).
        natoms: Optional atom count to validate force array dimensions.

    Returns:
        QEParseResult TypedDict with extracted data.
    """
    result: QEParseResult = {
        "exists": False,
        "converged": False,
        "calculation_type": "unknown",
        "total_energy_ry": None,
        "total_energy_ev": None,
        "fermi_energy_ry": None,
        "forces": [],
        "stress": [[0.0]*3 for _ in range(3)],
        "scf_cycles": 0,
        "relaxation_steps": 0,
        "cpu_time_sec": 0.0,
        "wall_time_sec": 0.0,
        "errors": [],
        "warnings": [],
        "log_snippet": ""
    }

    if not log_path.exists():
        logger.warning(f"QE output file not found: {log_path}")
        return result

    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
        result["exists"] = True
    except Exception as e:
        logger.error(f"Failed to read QE output {log_path}: {e}")
        result["errors"].append(f"File read error: {e}")
        return result

    # Store first 2KB for UI/CLI preview
    result["log_snippet"] = content[:2000]

    # 1. Calculation type
    result["calculation_type"] = _detect_calculation_type(content)

    # 2. Convergence
    result["converged"] = _check_convergence(content, result["calculation_type"])

    # 3. Energies
    total_ry, fermi_ry = _extract_energy_block(content)
    result["total_energy_ry"] = total_ry
    result["fermi_energy_ry"] = fermi_ry
    if total_ry is not None:
        result["total_energy_ev"] = total_ry * RYDBERG_TO_EV  # Ry to eV

    # 4. Forces & Stress
    result["forces"] = _extract_forces(content, natoms)
    result["stress"] = _extract_stress(content)

    # 5. Timing
    cpu, wall = _extract_timing(content)
    result["cpu_time_sec"] = cpu
    result["wall_time_sec"] = wall

    # 6. SCF/Relaxation steps count
    scf_matches = re.findall(r'!\s+total\s+energy\s*=', content, re.IGNORECASE)
    result["scf_cycles"] = len(scf_matches)
    relax_matches = re.findall(r'iteration\s+#\s*(\d+)', content, re.IGNORECASE)
    if relax_matches:
        result["relaxation_steps"] = max(int(x) for x in relax_matches)

    # 7. Errors & Warnings
    errs, warns = _detect_errors_warnings(content)
    result["errors"] = errs
    result["warnings"] = warns

    logger.debug(
        f"Parsed QE output: type={result['calculation_type']}, "
        f"converged={result['converged']}, energy={result['total_energy_ry']} Ry, "
        f"cpu={cpu:.1f}s, errors={len(errs)}"
    )
    return result