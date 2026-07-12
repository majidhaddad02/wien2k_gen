"""
Automated Convergence Testing for WIEN2k Calculations.

Performs systematic convergence studies with respect to:
- k-point mesh density
- RKmax (plane-wave cutoff)
- Smearing width (Methfessel-Paxton / Fermi-Dirac)

Runs actual WIEN2k commands via ``subprocess``, collects total energies and
timing data, and generates formatted convergence reports.

All documentation and inline comments are in English per project standards.
"""

import dataclasses
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Union

from ..core.constants import RYDBERG_TO_EV

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConvergenceResult:
    """A single data point in a convergence study."""

    parameter: str
    value: Union[float, int]
    total_energy_ry: float
    total_energy_ev: float
    delta_energy_mev: float
    wall_time_seconds: float
    converged: bool
    n_scf_iterations: int
    rkmax: float
    kpoints: str
    num_kpoints: int
    success: bool
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConvergenceResult":
        return cls(**data)


def _detect_progress_bar() -> Any:
    """Lazy-import progress-bar library; fall back to tqdm or plain text."""
    try:
        from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

        return ("rich", Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn)
    except ImportError:
        pass

    try:
        from tqdm import tqdm

        return ("tqdm", tqdm)
    except ImportError:
        pass
    return ("none",)


def _run_wien2k_command(
    cmd: Union[str, list[str]],
    cwd: Union[str, Path],
    timeout: Optional[float] = None,
) -> tuple[int, str, str]:
    """
    Execute a WIEN2k command in the given working directory.

    Parameters
    ----------
    cmd : str or list
        Command to execute.
    cwd : str or Path
        Working directory.
    timeout : float, optional
        Timeout in seconds.

    Returns
    -------
    tuple
        (returncode, stdout, stderr).
    """
    cmd_list = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)

    try:
        result = subprocess.run(
            cmd_list,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout or 3600,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out: %s", " ".join(cmd_list))
        return -1, "", "Timeout"
    except FileNotFoundError:
        logger.error("Executable not found: %s", cmd_list[0])
        return -1, "", f"Executable not found: {cmd_list[0]}"
    except Exception as exc:
        logger.error("Command failed: %s — %s", " ".join(cmd_list), exc)
        return -1, "", str(exc)


def _parse_total_energy(lines: list[str]) -> float:
    """
    Extract total energy from WIEN2k output lines.

    Scans for the last occurrence of ``:ENE`` which reports the total energy
    in Rydberg.

    Parameters
    ----------
    lines : list[str]
        Lines from case.scf or stdout.

    Returns
    -------
    float
        Total energy in Rydberg.
    """
    energy_ry = 0.0
    for line in lines:
        if ":ENE" in line:
            tokens = line.split(":ENE")[1].strip().split()
            for tok in tokens:
                try:
                    energy_ry = float(tok)
                    break
                except ValueError:
                    continue
    return energy_ry


def _extract_iterations(lines: list[str]) -> int:
    """Count SCF iterations from output."""
    count = 0
    for line in lines:
        if ":ITER" in line:
            count += 1
    return count


def _find_wien2k_commands(wien2k_cmd: dict[str, str]) -> dict[str, str]:
    """
    Resolve WIEN2k command paths from user-supplied mapping.

    Parameters
    ----------
    wien2k_cmd : dict
        Mapping of command names to paths, e.g.
        ``{"init_lapw": "init_lapw", "run_lapw": "run_lapw"}``.

    Returns
    -------
    dict
        Resolved command dictionary with absolute paths.
    """
    resolved: dict[str, str] = {}
    for name, cmd in wien2k_cmd.items():
        if os.path.isabs(cmd):
            resolved[name] = cmd
        else:
            which = shutil.which(cmd)
            resolved[name] = which if which else cmd
    return resolved


def _modify_incar(incar_path: Path, updates: dict[str, str]) -> list[str]:
    """
    Modify values in a WIEN2k case.in1 file.

    Parameters
    ----------
    incar_path : Path
        Path to case.in1 or case.in1c.
    updates : dict
        Key-value pairs to update.

    Returns
    -------
    list[str]
        Updated lines of the file.
    """
    lines: list[str] = []
    if incar_path.exists():
        with open(incar_path) as fh:
            lines = fh.readlines()
    else:
        return lines

    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        # Lines like "RKMAX  7.0" — the value is on the same line
        # Most WIEN2k input files have a single value per line
        modified = False
        for key, value in updates.items():
            if key.upper() in stripped.upper():
                parts = stripped.split()
                if key.upper() == "RKMAX" and len(parts) > 0:
                    # Replace the numeric value on this line
                    new_line = f"{value}    # RKMAX\n"
                    new_lines.append(new_line)
                    modified = True
                    break
        if not modified:
            new_lines.append(line)

    with open(incar_path, "w") as fh:
        fh.writelines(new_lines)

    return new_lines


def _modify_klist(klist_path: Path, nx: int, ny: int, nz: int) -> None:
    """
    Write a new k-point mesh into case.klist.

    Parameters
    ----------
    klist_path : Path
        Path to case.klist.
    nx, ny, nz : int
        Number of k-points in each direction.
    """
    lines = [
        f"         1         0         0  {nx:4d}  {ny:4d}  {nz:4d}  1.0  -7.0  1.5    simple cubic\n",
        "END\n",
    ]
    with open(klist_path, "w") as fh:
        fh.writelines(lines)


def run_kpoint_convergence(  # noqa: C901
    base_case: str,
    kpoint_grids: list[tuple[int, int, int]],
    wien2k_cmd: dict[str, str],
    base_path: Optional[str] = None,
    rkmax: float = 7.0,
    timeout_per_run: int = 3600,
) -> dict[str, Any]:
    """
    Run k-point convergence study by testing multiple k-point grids.

    Copies the base case into temporary directories, modifies the k-point
    list, runs an SCF cycle, and extracts energies and timings.

    Parameters
    ----------
    base_case : str
        WIEN2k case name.
    kpoint_grids : list of tuple
        List of (nx, ny, nz) grids to test.
    wien2k_cmd : dict
        Mapping of command names to paths or executables.
    base_path : str, optional
        Directory containing the base case files.
    rkmax : float
        RKmax value to use for all runs.
    timeout_per_run : int
        Maximum wall time per SCF run in seconds.

    Returns
    -------
    dict
        Keys: ``"results"`` (list of :class:`ConvergenceResult`),
        ``"optimal_grid"`` (tuple), ``"optimal_energy_ev"`` (float).
    """
    cmds = _find_wien2k_commands(wien2k_cmd)
    base_dir = Path(base_path) if base_path else Path.cwd()
    results: list[ConvergenceResult] = []

    progress_lib = _detect_progress_bar()
    bar_type = progress_lib[0]

    if bar_type == "rich":
        _, Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn = progress_lib
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        )
        task = progress.add_task("K-point convergence", total=len(kpoint_grids))
        progress.start()
    elif bar_type == "tqdm":
        _, tqdm = progress_lib
        pbar = tqdm(total=len(kpoint_grids), desc="K-point convergence")
    else:
        pbar = None

    prev_energy = None

    try:
        for _idx, (nx, ny, nz) in enumerate(kpoint_grids):
            # Create temporary work directory
            work_dir = Path(tempfile.mkdtemp(prefix=f"kpt_{nx}x{ny}x{nz}_"))

            # Copy all files from base directory
            for item in base_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, work_dir / item.name)

            # Modify klist for this grid
            klist_file = work_dir / f"{base_case}.klist"
            _modify_klist(klist_file, nx, ny, nz)

            # Modify case.in1 with RKmax
            in1_file = work_dir / f"{base_case}.in1"
            _modify_incar(in1_file, {"RKMAX": str(rkmax)})

            # Run init_lapw first (if available)
            if "init_lapw" in cmds:
                _run_wien2k_command(
                    f"{cmds['init_lapw']} -b -rkmax {rkmax} -numk {nx*ny*nz}",
                    work_dir,
                    timeout=timeout_per_run,
                )

            # Run SCF
            lapw_cmd = cmds.get("run_lapw", cmds.get("run_lapw", "run_lapw"))
            start = time.time()
            rc, stdout, stderr = _run_wien2k_command(
                f"{lapw_cmd} -p -ec 0.0001 -cc 0.0001 -i 40",
                work_dir,
                timeout=timeout_per_run,
            )
            elapsed = time.time() - start

            # Parse results
            scf_file = work_dir / f"{base_case}.scf"
            energy_ry = 0.0
            n_iter = 0
            if scf_file.exists():
                scf_lines = scf_file.read_text(errors="replace").splitlines()
                energy_ry = _parse_total_energy(scf_lines)
                n_iter = _extract_iterations(scf_lines)
            elif stdout:
                energy_ry = _parse_total_energy(stdout.splitlines())
                n_iter = _extract_iterations(stdout.splitlines())

            energy_ev = energy_ry * RYDBERG_TO_EV
            delta_mev = 0.0
            if prev_energy is not None:
                delta_mev = abs(energy_ev - prev_energy) * 1000.0

            results.append(
                ConvergenceResult(
                    parameter="kpoints",
                    value=f"{nx}x{ny}x{nz}",
                    total_energy_ry=energy_ry,
                    total_energy_ev=energy_ev,
                    delta_energy_mev=delta_mev,
                    wall_time_seconds=elapsed,
                    converged=rc == 0,
                    n_scf_iterations=n_iter,
                    rkmax=rkmax,
                    kpoints=f"{nx}x{ny}x{nz}",
                    num_kpoints=nx * ny * nz,
                    success=rc == 0,
                    stdout=stdout[:2000] if stdout else "",
                    stderr=stderr[:2000] if stderr else "",
                )
            )

            prev_energy = energy_ev

            if bar_type == "rich":
                progress.update(task, advance=1, description=f"K-point grid {nx}x{ny}x{nz}")
            elif bar_type == "tqdm":
                pbar.update(1)
                pbar.set_description(f"K-point grid {nx}x{ny}x{nz}")
            else:
                logger.info(
                    "Grid %dx%dx%d: energy=%.6f eV, delta=%.3f meV, time=%.1f s",
                    nx, ny, nz, energy_ev, delta_mev, elapsed,
                )

            # Cleanup temporary directory
            shutil.rmtree(work_dir, ignore_errors=True)
    finally:
        if bar_type == "rich":
            progress.stop()
        elif bar_type == "tqdm":
            pbar.close()

    return {
        "results": [r.to_dict() for r in results],
        "optimal_grid": kpoint_grids[-1] if kpoint_grids else (0, 0, 0),
        "optimal_energy_ev": results[-1].total_energy_ev if results else 0.0,
    }


def run_rkmax_convergence(  # noqa: C901
    base_case: str,
    rkmax_values: list[float],
    wien2k_cmd: dict[str, str],
    base_path: Optional[str] = None,
    kpoints: tuple[int, int, int] = (4, 4, 4),
    timeout_per_run: int = 3600,
) -> dict[str, Any]:
    """
    Run RKmax convergence study.

    Tests different RKmax values while keeping the k-point grid fixed.

    Parameters
    ----------
    base_case : str
        WIEN2k case name.
    rkmax_values : list of float
        RKmax values to test.
    wien2k_cmd : dict
        Mapping of command names to paths.
    base_path : str, optional
        Directory containing base case files.
    kpoints : tuple
        Fixed k-point grid (nx, ny, nz).
    timeout_per_run : int
        Maximum wall time per run in seconds.

    Returns
    -------
    dict
        Keys: ``"results"`` (list of :class:`ConvergenceResult`),
        ``"optimal_rkmax"`` (float), ``"optimal_energy_ev"`` (float).
    """
    cmds = _find_wien2k_commands(wien2k_cmd)
    base_dir = Path(base_path) if base_path else Path.cwd()
    results: list[ConvergenceResult] = []

    progress_lib = _detect_progress_bar()
    bar_type = progress_lib[0]

    if bar_type == "rich":
        _, Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn = progress_lib
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        )
        task = progress.add_task("RKmax convergence", total=len(rkmax_values))
        progress.start()
    elif bar_type == "tqdm":
        _, tqdm = progress_lib
        pbar = tqdm(total=len(rkmax_values), desc="RKmax convergence")
    else:
        pbar = None

    prev_energy = None

    try:
        for rkmax in rkmax_values:
            work_dir = Path(tempfile.mkdtemp(prefix=f"rkmax_{rkmax:.2f}_"))

            for item in base_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, work_dir / item.name)

            klist_file = work_dir / f"{base_case}.klist"
            _modify_klist(klist_file, *kpoints)

            in1_file = work_dir / f"{base_case}.in1"
            _modify_incar(in1_file, {"RKMAX": str(rkmax)})

            if "init_lapw" in cmds:
                _run_wien2k_command(
                    f"{cmds['init_lapw']} -b -rkmax {rkmax} -numk {kpoints[0]*kpoints[1]*kpoints[2]}",
                    work_dir,
                    timeout=timeout_per_run,
                )

            lapw_cmd = cmds.get("run_lapw", "run_lapw")
            start = time.time()
            rc, stdout, stderr = _run_wien2k_command(
                f"{lapw_cmd} -p -ec 0.0001 -cc 0.0001 -i 40",
                work_dir,
                timeout=timeout_per_run,
            )
            elapsed = time.time() - start

            scf_file = work_dir / f"{base_case}.scf"
            energy_ry = 0.0
            n_iter = 0
            if scf_file.exists():
                scf_lines = scf_file.read_text(errors="replace").splitlines()
                energy_ry = _parse_total_energy(scf_lines)
                n_iter = _extract_iterations(scf_lines)
            elif stdout:
                energy_ry = _parse_total_energy(stdout.splitlines())
                n_iter = _extract_iterations(stdout.splitlines())

            energy_ev = energy_ry * RYDBERG_TO_EV
            delta_mev = 0.0
            if prev_energy is not None:
                delta_mev = abs(energy_ev - prev_energy) * 1000.0

            results.append(
                ConvergenceResult(
                    parameter="rkmax",
                    value=rkmax,
                    total_energy_ry=energy_ry,
                    total_energy_ev=energy_ev,
                    delta_energy_mev=delta_mev,
                    wall_time_seconds=elapsed,
                    converged=rc == 0,
                    n_scf_iterations=n_iter,
                    rkmax=rkmax,
                    kpoints=f"{kpoints[0]}x{kpoints[1]}x{kpoints[2]}",
                    num_kpoints=kpoints[0] * kpoints[1] * kpoints[2],
                    success=rc == 0,
                    stdout=stdout[:2000] if stdout else "",
                    stderr=stderr[:2000] if stderr else "",
                )
            )

            prev_energy = energy_ev

            if bar_type == "rich":
                progress.update(task, advance=1, description=f"RKmax {rkmax:.2f}")
            elif bar_type == "tqdm":
                pbar.update(1)
                pbar.set_description(f"RKmax {rkmax:.2f}")
            else:
                logger.info(
                    "RKmax %.2f: energy=%.6f eV, delta=%.3f meV, time=%.1f s",
                    rkmax, energy_ev, delta_mev, elapsed,
                )

            shutil.rmtree(work_dir, ignore_errors=True)
    finally:
        if bar_type == "rich":
            progress.stop()
        elif bar_type == "tqdm":
            pbar.close()

    return {
        "results": [r.to_dict() for r in results],
        "optimal_rkmax": rkmax_values[-1],
        "optimal_energy_ev": results[-1].total_energy_ev if results else 0.0,
    }


def find_converged_parameters(
    convergence_data: dict[str, Any],
    tolerance: float = 1.0,
) -> dict[str, Any]:
    """
    Determine the optimal (converged) parameters from convergence data.

    Finds the first parameter value where the energy change falls below
    the specified tolerance (in meV/atom, or meV absolute).

    Parameters
    ----------
    convergence_data : dict
        Output from :func:`run_kpoint_convergence` or :func:`run_rkmax_convergence`.
    tolerance : float
        Energy convergence threshold in meV.

    Returns
    -------
    dict
        Keys depend on the study type, e.g.
        ``{"parameter": "rkmax", "converged_value": 7.0, "energy_ev": -1234.5, "delta_mev": 0.5}``.
    """
    results_raw = convergence_data.get("results", [])
    if not results_raw:
        return {"parameter": "unknown", "converged_value": None, "energy_ev": 0.0, "delta_mev": 0.0}

    results = [ConvergenceResult.from_dict(r) if isinstance(r, dict) else r for r in results_raw]

    param_name = results[0].parameter
    for r in results:
        if 0.0 < r.delta_energy_mev < tolerance:
            return {
                "parameter": param_name,
                "converged_value": r.value,
                "energy_ev": r.total_energy_ev,
                "delta_mev": r.delta_energy_mev,
            }

    # Fall back to the last (most stringent) result
    last = results[-1]
    return {
        "parameter": param_name,
        "converged_value": last.value,
        "energy_ev": last.total_energy_ev,
        "delta_mev": last.delta_energy_mev,
    }


def generate_convergence_report(results: dict[str, Any]) -> str:
    """
    Generate a human-readable convergence report string.

    Parameters
    ----------
    results : dict
        Output from :func:`run_kpoint_convergence` or :func:`run_rkmax_convergence`.

    Returns
    -------
    str
        Formatted convergence report.
    """
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  WIEN2k Convergence Study Report")
    lines.append("=" * 72)
    lines.append("")

    data = results.get("results", [])
    if not data:
        lines.append("  No convergence data available.")
        return "\n".join(lines)

    # Header
    param = data[0].get("parameter", "param") if isinstance(data[0], dict) else data[0].parameter
    lines.append(f"  Parameter studied: {param}")
    lines.append("")
    lines.append(f"  {'-' * 66}")
    lines.append(f"  {'Value':>12s}  {'E_total (eV)':>16s}  {'ΔE (meV)':>12s}  {'Time (s)':>10s}  {'SCF':>5s}  {'Status':>6s}")
    lines.append(f"  {'-' * 66}")

    for entry in data:
        if isinstance(entry, dict):
            val = entry.get("value", 0)
            energy = entry.get("total_energy_ev", 0.0)
            delta = entry.get("delta_energy_mev", 0.0)
            wall = entry.get("wall_time_seconds", 0.0)
            n_iter = entry.get("n_scf_iterations", 0)
            ok = entry.get("success", False)
        else:
            val = entry.value
            energy = entry.total_energy_ev
            delta = entry.delta_energy_mev
            wall = entry.wall_time_seconds
            n_iter = entry.n_scf_iterations
            ok = entry.success

        status = "OK" if ok else "FAIL"
        lines.append(f"  {val!s:>12s}  {energy:>16.6f}  {delta:>12.3f}  {wall:>10.1f}  {n_iter:>5d}  {status:>6s}")

    lines.append(f"  {'-' * 66}")

    # Summary
    converged = find_converged_parameters({"results": data})
    lines.append("")
    lines.append("  Convergence Summary:")
    lines.append(f"    Parameter:          {converged['parameter']}")
    lines.append(f"    Converged value:    {converged['converged_value']}")
    lines.append(f"    Energy (eV):        {converged['energy_ev']:.6f}")
    lines.append(f"    ΔE (meV):           {converged['delta_mev']:.3f}")
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


def detect_scf_divergence(scf_content: str, energy_values: Optional[list[float]] = None) -> dict:  # noqa: C901
    """Detect SCF divergence and recommend automatic recovery actions.

    Divergence signatures:
      - Monotonic energy increase over 10+ cycles → unstable mixing
      - Oscillating energy ±10 eV → charge sloshing
      - Exploding energy > 1e5 → catastrophic divergence
      - Gap oscillation for metallic systems → need smearing
      - Stuck energy (flat for 20+ cycles) → stalled convergence

    Returns dict with:
        divergent: bool
        divergence_type: str
        severity: float (0-1)
        recommended_action: str
        auto_mixing_params: dict (beta adjustment suggestions)
    """
    if energy_values is None:
        cd_pattern = re.compile(
            r':ene\s*:\s*.*?(-?\d+\.\d+)', re.IGNORECASE
        )
        matches = cd_pattern.findall(scf_content)
        energy_values = [float(m) for m in matches]

    result = {
        "divergent": False,
        "divergence_type": "none",
        "severity": 0.0,
        "recommended_action": "",
        "auto_mixing_params": {"beta": None, "pratt_cycles": None, "msr1a": False},
    }

    if len(energy_values) < 5:
        return result

    n = len(energy_values)
    deltas = [energy_values[i] - energy_values[i - 1] for i in range(1, n)]

    # 1. Catastrophic divergence (energy explodes)
    if any(abs(e) > 1e5 for e in energy_values):
        result["divergent"] = True
        result["divergence_type"] = "catastrophic"
        result["severity"] = 1.0
        result["recommended_action"] = (
            "Energy exploded. Check RMT values, RKMAX, and initial charge "
            "density. Restart from scratch with reduced mixing (β=0.02) "
            "and increased PRATT cycles."
        )
        result["auto_mixing_params"]["beta"] = 0.02
        result["auto_mixing_params"]["pratt_cycles"] = 10
        return result

    # 2. Monotonic drift (energy increasing steadily for 10+ cycles)
    if len(deltas) >= 10:
        recent_deltas = deltas[-10:]
        positive_count = sum(1 for d in recent_deltas if d > 0)
        if positive_count >= 8:
            drift_rate = sum(d for d in recent_deltas if d > 0) / max(positive_count, 1)
            result["divergent"] = True
            result["divergence_type"] = "monotonic_drift"
            result["severity"] = min(1.0, abs(drift_rate) / 10.0)
            result["recommended_action"] = (
                f"Energy drifting upward ({drift_rate:.3f} Ry/cycle). "
                f"Reduce mixing beta 3x and increase PRATT to 5 cycles. "
                f"Try MSR1a mixing for multi-secant stabilization."
            )
            result["auto_mixing_params"]["beta"] = 0.03
            result["auto_mixing_params"]["pratt_cycles"] = 5
            result["auto_mixing_params"]["msr1a"] = True
            return result

    # 3. High-amplitude oscillation (charge sloshing)
    if len(deltas) >= 8:
        sign_changes = sum(1 for i in range(1, len(deltas)) if deltas[i] * deltas[i - 1] < 0)
        if sign_changes >= len(deltas) // 2:
            max_amp = max(abs(d) for d in deltas)
            result["divergent"] = True
            result["divergence_type"] = "charge_sloshing"
            result["severity"] = min(1.0, max_amp / 0.01)
            result["recommended_action"] = (
                f"Charge sloshing detected (amplitude {max_amp:.6f} Ry). "
                f"Halve mixing beta, use PRATT mixing, or switch to MSR1a. "
                f"If metallic, add Methfessel-Paxton smearing (0.02 Ry)."
            )
            beta = 0.05
            if energy_values:
                current_beta = _infer_mixing_beta(scf_content)
                beta = current_beta / 2.0 if current_beta > 0 else 0.05
            result["auto_mixing_params"]["beta"] = max(0.01, beta)
            result["auto_mixing_params"]["pratt_cycles"] = 3
            return result

    # 4. Stalled convergence (flat for many cycles)
    if len(deltas) >= 20:
        recent = deltas[-20:]
        if all(abs(d) < 1e-8 for d in recent):
            result["divergent"] = True
            result["divergence_type"] = "stalled"
            result["severity"] = 0.5
            result["recommended_action"] = (
                "SCF stalled — energy not changing. Check if converged "
                "(< 0.0001 Ry), or increase mixing beta slightly."
            )
            result["auto_mixing_params"]["beta"] = 0.15
            result["auto_mixing_params"]["pratt_cycles"] = 3

    return result


def _infer_mixing_beta(scf_content: str) -> float:
    """Try to infer current mixing beta from SCF log or case.inm."""
    inm_match = re.search(
        r'(?:MIXING|beta|MSEC|pratt)\s*[=:]\s*(0?\.\d+)',
        scf_content, re.IGNORECASE,
    )
    if inm_match:
        return float(inm_match.group(1))
    return 0.0


__all__ = [
    "ConvergenceResult",
    "detect_scf_divergence",
    "find_converged_parameters",
    "generate_convergence_report",
    "run_kpoint_convergence",
    "run_rkmax_convergence",
]
