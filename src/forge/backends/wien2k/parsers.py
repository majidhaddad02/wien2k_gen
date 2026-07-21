"""WIEN2k file parsers — extract problem parameters from DFT input/output files."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from ...types import Wien2kFlags

from ...logging_config import get_logger

logger = get_logger(__name__)


class DayfileResult(TypedDict, total=False):
    """Structured output for dayfile parsing."""
    exists: bool
    times: dict[str, float]
    bottleneck: str | None
    errors: list[str]
    warnings: list[str]
    convergence: str | None
    cycles_completed: int


class OutputParseResult(TypedDict, total=False):
    """Structured output for general log parsing."""
    exists: bool
    converged: bool | None
    errors: list[str]
    timing: dict[str, float]
    content_snippet: str


def parse_output(log_path: Path) -> dict[str, Any]:
    """Parse WIEN2k output files for convergence and errors."""
    if log_path.suffix == ".dayfile" or "dayfile" in log_path.name:
        return parse_dayfile(str(log_path))

    if not log_path.exists():
        return {"exists": False, "converged": None, "errors": [], "timing": {}}

    try:
        content = log_path.read_text(encoding="utf-8", errors="replace").lower()
        converged = any(phrase in content for phrase in [
            "charge convergence", "energy convergence", "scf cycle converged"
        ])

        timing = {}
        time_pattern = r"(\w+):\s+cpu time:\s+([\d\.]+)"
        for match in re.finditer(time_pattern, content, re.IGNORECASE):
            prog = match.group(1).lower()
            timing[prog] = float(match.group(2))

        errors = []
        error_patterns = {
            "qtl-b": "QTL-B error: check case.in1 and convergence parameters",
            "lapw": "LAPWx crash: check MPI communication and memory limits",
            "error while loading shared libraries": "Missing library: check LD_LIBRARY_PATH and WIENROOT",
            "segmentation fault": "Segmentation fault: check memory limits and array bounds",
        }
        for pattern, msg in error_patterns.items():
            if pattern in content:
                errors.append(msg)

        return {
            "exists": True,
            "converged": converged,
            "errors": errors,
            "timing": timing,
            "content_snippet": content[:1000] if len(content) > 1000 else content
        }
    except Exception as e:
        logger.warning(f"Could not parse output {log_path}: {e}")
        return {"exists": True, "converged": None, "errors": [f"Parse error: {e}"], "timing": {}}


def parse_dayfile(dayfile_path: str = "case.dayfile") -> DayfileResult:  # noqa: C901
    """
    Parse WIEN2k .dayfile for timing, bottleneck detection, and error analysis.
    Returns structured data for performance monitoring and auto-tuning.
    """
    result: DayfileResult = {
        "exists": False,
        "times": {"lapw0": 0.0, "lapw1": 0.0, "lapw2": 0.0, "lapwso": 0.0, "mixer": 0.0},
        "bottleneck": None,
        "errors": [],
        "warnings": [],
        "convergence": None,
        "cycles_completed": 0,
    }

    path = Path(dayfile_path)
    if not path.exists():
        dayfiles = sorted(Path(".").glob("*.dayfile"), key=lambda p: p.stat().st_mtime, reverse=True)
        if dayfiles:
            path = dayfiles[0]
        else:
            return result

    result["exists"] = True

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Could not read dayfile {path}: {e}")
        return result

    # Extract timing for each program
    time_pattern = r"(\w+):\s+starting at\s+\S+\s+ended at\s+\S+\s+.*?cpu time:\s+([\d\.]+)"
    for match in re.finditer(time_pattern, content, re.IGNORECASE):
        prog = match.group(1).lower()
        time_val = float(match.group(2))
        for key in result["times"]:
            if key in prog or prog in key:
                result["times"][key] = time_val
                break

    # Detect bottleneck
    times = {k: v for k, v in result["times"].items() if v > 0}
    if times:
        max_prog = max(times, key=times.get)
        max_time = times[max_prog]
        total_time = sum(times.values())

        if max_time > total_time * 0.6 and max_time > 10.0:
            result["bottleneck"] = max_prog
            if max_prog == "lapw2":
                result["warnings"].append(
                    "lapw2 is the bottleneck. Consider enabling lapw2_vector_split or reducing cores per node."
                )
            elif max_prog == "lapw1":
                result["warnings"].append(
                    "lapw1 is the bottleneck. Consider increasing k-point parallelization (kpar)."
                )

    # Convergence status
    lower_content = content.lower()
    if "charge convergence" in lower_content or "energy convergence" in lower_content:
        result["convergence"] = "converged"
    elif "not converged" in lower_content or "diverged" in lower_content:
        result["convergence"] = "not_converged"
        result["warnings"].append("SCF did not converge. Check mixing parameters and case.in1.")

    # Count completed cycles
    cycle_matches = re.findall(r"cycle\s+(\d+)", content, re.IGNORECASE)
    if cycle_matches:
        result["cycles_completed"] = max(int(c) for c in cycle_matches)

    # Detect common errors
    error_patterns = {
        "QTL-B": "QTL-B error: check case.in1, RKMAX, and convergence parameters",
        "LAPWx crashed": "LAPWx crashed: check MPI communication, memory limits, and case.struct",
        "error while loading shared libraries": "Missing shared library: check LD_LIBRARY_PATH and WIENROOT",
        "segmentation fault": "Segmentation fault: check memory limits and array bounds",
        "MPI_ABORT": "MPI abort: check network connectivity and process placement",
    }
    for pattern, msg in error_patterns.items():
        if pattern in content:
            result["errors"].append(msg)

    return result


def detect_problem_size() -> dict[str, Any]:  # noqa: C901
    """
    Extract problem parameters from WIEN2k input files.
    Uses CaseFileParser (preferred) with integrated fallback to legacy parsing.

    References:
        Blaha et al. (2020) J. Chem. Phys. 152, 074101 (WIEN2k Usersguide Sec. 4.1-4.5)
        Cebrián et al. (2015) Comput. Phys. Commun. 201, 85-99
    """
    # Try CaseFileParser first for LDA+U parameters and modern parsing
    try:
        from ...core.case_parser import CaseFileParser as _CFP
        parser = _CFP()
        case_data = parser.parse_all()

        result: dict[str, Any] = {
            "atoms": case_data.atoms,
            "kpoints": case_data.kpoints,
            "nmat": case_data.nmat,
            "nbands": case_data.nbands,
            "rkmax": case_data.rkmax,
            "is_soc": case_data.is_soc,
            "is_hybrid": case_data.is_hybrid,
            "is_spin_polarized": case_data.is_spin_polarized,
            "is_lda_u": case_data.is_lda_u,
            "is_eece": case_data.is_eece,
            "has_forces": case_data.has_forces,
            "complexity": 1.0,
            # NEW: LDA+U parameters from .inm
            "_ldau_u_ry": case_data.ldau.u_ry,
            "_ldau_j_ry": case_data.ldau.j_ry,
            "_ldau_ueff_ry": case_data.ldau.ueff_ry,
            "_ldau_dc": case_data.ldau.double_counting,
            "_fft_nx": case_data.fft_nx,
            "_fft_ny": case_data.fft_ny,
            "_fft_nz": case_data.fft_nz,
            "_gmax": case_data.gmax,
        }

        if result["atoms"] == 0 and result["nmat"] == 0:
            result["atoms"] = 10
        if result["complexity"] == 1.0 and result["atoms"] > 0:
            result["complexity"] = result["atoms"] / 50.0

        # Still call detect_wien2k_flags for calc_type and exec_command
        flags = detect_wien2k_flags()
        result["is_spin_polarized"] = result["is_spin_polarized"] or flags.is_spin_polarized
        result["is_lda_u"] = result["is_lda_u"] or flags.is_lda_u
        result["is_eece"] = result["is_eece"] or flags.is_eece
        result["has_forces"] = result["has_forces"] or flags.has_forces
        if not result.get("is_soc"):
            result["is_soc"] = flags.is_soc
        if not result.get("is_hybrid"):
            result["is_hybrid"] = flags.is_hybrid
        result["calc_type"] = flags.get_calculation_type().value
        result["exec_command"] = flags.get_execution_command()

        return result
    except Exception:
        logger.debug("Suppressed exception in detect_problem_size()", exc_info=True)

    # Fallback to legacy parsing for robustness
    result: dict[str, Any] = {
        "atoms": 10, "kpoints": 0, "nmat": 0, "nbands": None,
        "rkmax": 7.0, "is_soc": False, "is_hybrid": False, "complexity": 1.0
    }

    # 1. Extract atoms from .struct file
    # Standard WIEN2k .struct format:
    #   Line 1: Title
    #   Line 2: LATTICE,NONEQUIV.ATOMS: N  SPACEGROUP
    #   Line 3: MODE OF CALC=... unit=...
    #   Line 4: a b c alpha beta gamma
    #   For each inequivalent atom: ATOM line, MULT line, element line, rotation matrix
    struct_files = list(Path(".").glob("*.struct"))
    if struct_files:
        try:
            content = struct_files[0].read_text(encoding="utf-8", errors="replace")
            # Primary: parse NONEQUIV.ATOMS from line 2 (number of inequivalent atoms)
            match = re.search(r'NONEQUIV\.ATOMS\s*:\s*(\d+)', content, re.IGNORECASE)
            if match:
                nat_inequiv = int(match.group(1))
                # Try to sum multiplicities for total atom count
                mult_matches = re.findall(r'MULT\s*=\s*(\d+)', content, re.IGNORECASE)
                if mult_matches and len(mult_matches) >= nat_inequiv:
                    result["atoms"] = sum(int(m) for m in mult_matches[:nat_inequiv])
                else:
                    result["atoms"] = nat_inequiv
            else:
                # Fallback 1: sum MULT values
                mult_matches = re.findall(r'MULT\s*=\s*(\d+)', content, re.IGNORECASE)
                if mult_matches:
                    result["atoms"] = sum(int(m) for m in mult_matches)
                else:
                    # Fallback 2: count ATOM lines
                    atom_lines = [line for line in content.splitlines() if re.match(r'^\s*ATOM\s*[-\d]+:', line, re.IGNORECASE)]
                    if atom_lines:
                        result["atoms"] = len(atom_lines)
                    else:
                        # Fallback 3: count coordinate-like lines
                        coord_lines = sum(
                            1 for line in content.splitlines()
                            if re.match(r'^\s*[A-Za-z][A-Za-z0-9]?\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+\s+[-+]?\d*\.\d+', line)
                        )
                        if coord_lines > 0:
                            result["atoms"] = coord_lines
        except Exception as e:
            logger.warning(f"Failed to parse .struct file: {e}")

    # 2. Extract k-points from .klist
    # Format: first line is k-point count, or count non-empty lines minus header
    klist_files = list(Path(".").glob("*.klist*"))
    if klist_files:
        try:
            content = klist_files[0].read_text(encoding="utf-8", errors="replace")
            lines = [line.strip() for line in content.splitlines() if line.strip()
                     and not line.strip().startswith("#")]
            # First line typically contains k-point count or header
            first_line = lines[0] if lines else ""
            parts = first_line.split()
            if parts and parts[0].isdigit():
                result["kpoints"] = int(parts[0])
            elif len(lines) > 1:
                # Fallback: count data lines (each k-point has weight + coordinates)
                result["kpoints"] = len(lines)
        except Exception as e:
            logger.debug(f"Could not parse kpoints from .klist: {e}")

    # 3. Extract nmat from .scf file (exact value from SCF run)
    scf_files = list(Path(".").glob("*.scf"))
    if scf_files:
        try:
            content = scf_files[0].read_text(encoding="utf-8", errors="replace")
            match = re.search(r':NMAT\s+(\d+)', content)
            if match:
                result["nmat"] = int(match.group(1))
        except Exception as e:
            logger.debug(f"Could not parse nmat from .scf: {e}")

    # 3b. Estimate nmat from .in2 FFT grid (fallback when .scf doesn't exist)
    if result["nmat"] == 0:
        in2_files = list(Path(".").glob("*.in2*"))
        if in2_files:
            try:
                content = in2_files[0].read_text(encoding="utf-8", errors="replace")
                # .in2 line 3: NX NY NZ enhancement_factor iprint
                for line in content.splitlines():
                    stripped = line.strip()
                    parts = stripped.split()
                    if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
                        nx, ny, nz = int(parts[0]), int(parts[1]), int(parts[2])
                        # nmat ≈ (FFT grid total) / fudge_factor
                        # For lapw1, nmat = G_max sphere within FFT box
                        fft_total = nx * ny * nz
                        estimated_nmat = int((fft_total ** (1.0 / 3.0)) * 1.1)
                        result["nmat"] = max(100, estimated_nmat)
                        break
            except Exception as e:
                logger.debug(f"Could not estimate nmat from .in2: {e}")

    # 4. Extract nbands from .in1 file
    # .in1 format:
    #   Line 1: WFFIL (or TOT for older versions)
    #   Line 2: RKMAX LMAX V-NMT
    #   Line 3: global E-param
    #   Line 4+: per-l quantum numbers
    in1_files = list(Path(".").glob("*.in1*"))
    if in1_files:
        try:
            for line in in1_files[0].read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                # Look for TOT keyword (old format) or count bands from
                # WFFIL mode which sets nbands implicitly from the case.vector
                if 'TOT' in stripped.upper():
                    parts = stripped.split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        result["nbands"] = int(parts[0])
                        break
            # Fallback: estimate from nmat
            if result["nbands"] is None and result["nmat"] > 0:
                result["nbands"] = max(10, result["nmat"] // 2)
        except Exception as e:
            logger.debug(f"Could not parse nbands from .in1: {e}")

    # 5. Detect SOC from .inso file
    if list(Path(".").glob("*.inso")):
        result["is_soc"] = True

    # 6. Detect hybrid functional from .in0 / .in0_st / .inc files
    # WIEN2k hybrid functionals (HSE, PBE0, etc.) are flagged by HYBR keyword
    # in .in0 or .in0_st.  The .inm file is for LDA+U (Hubbard U), NOT hybrid.
    hybrid_detected = False
    for hybrid_file_pat in ["*.in0", "*.in0_st", "*.in0_grr", "*.inc"]:
        hybrid_files = list(Path(".").glob(hybrid_file_pat))
        for hf in hybrid_files[:1]:
            try:
                content = hf.read_text(encoding="utf-8", errors="replace")
                if re.search(r'\bHYBR', content, re.IGNORECASE):
                    hybrid_detected = True
                    break
            except Exception:
                logger.debug("Suppressed exception in detect_problem_size()", exc_info=True)
        if hybrid_detected:
            break
    result["is_hybrid"] = hybrid_detected

    # 7. Extract RKMAX from .in0 file
    in0_files = list(Path(".").glob("*.in0"))
    if in0_files:
        try:
            content = in0_files[0].read_text(encoding="utf-8", errors="replace")
            match = re.search(r'RKMAX\s*=\s*([\d.]+)', content)
            if match:
                result["rkmax"] = float(match.group(1))
        except Exception as e:
            logger.debug(f"Could not parse RKMAX from .in0: {e}")

    # 8. Estimate complexity
    result["complexity"] = result["atoms"] / 50.0

    # 9. Detect WIEN2k flags (spin, SOC, LDA+U, hybrid, EECE, forces)
    flags = detect_wien2k_flags()
    result["is_spin_polarized"] = flags.is_spin_polarized
    result["is_lda_u"] = flags.is_lda_u
    result["is_eece"] = flags.is_eece
    result["has_forces"] = flags.has_forces
    if not result.get("is_soc"):
        result["is_soc"] = flags.is_soc
    if not result.get("is_hybrid"):
        result["is_hybrid"] = flags.is_hybrid
    result["calc_type"] = flags.get_calculation_type().value
    result["exec_command"] = flags.get_execution_command()

    return result


def detect_wien2k_flags() -> Wien2kFlags:  # noqa: C901
    """
    Detect WIEN2k calculation flags from input files.
    Determines the correct execution command and parallelization adjustments.

    Detection logic based on WIEN2k Usersguide (Blaha et al., 2020), Sections 4.1-4.4:
    - case.inst: Spin polarization (contains spin-up/down occupation strings)
    - case.inso: Spin-orbit coupling
    - case.inorb: LDA+U
    - case.in0 / case.in0_st: Hybrid functional (HYBR keyword)
    - case.ineece: Onsite exact exchange
    """
    from ...types import Wien2kFlags

    flags = Wien2kFlags()

    inst_files = list(Path(".").glob("*.inst"))
    if inst_files:
        try:
            content = inst_files[0].read_text(encoding="utf-8", errors="replace")
            flags.is_spin_polarized = "SPIN" in content.upper()
        except Exception:
            logger.debug("Suppressed exception in detect_wien2k_flags()", exc_info=True)

    if list(Path(".").glob("*.inso")):
        flags.is_soc = True

    if list(Path(".").glob("*.inorb")):
        flags.is_lda_u = True

    for hf_pat in ["*.in0", "*.in0_st", "*.in0_grr"]:
        for hf in list(Path(".").glob(hf_pat))[:1]:
            try:
                content = hf.read_text(encoding="utf-8", errors="replace")
                if re.search(r'\bHYBR', content, re.IGNORECASE):
                    flags.is_hybrid = True
                    break
            except Exception:
                logger.debug("Suppressed exception in detect_wien2k_flags()", exc_info=True)

    if list(Path(".").glob("*.ineece")):
        flags.is_eece = True

    wienroot = os.environ.get("WIENROOT")
    if wienroot:
        version_file = Path(wienroot, "VERSION")
        if version_file.exists():
            try:
                ver_str = version_file.read_text().strip().split()[0]
                major_minor = ".".join(ver_str.split(".")[:2]) if "." in ver_str else ver_str
                flags.wien2k_version = major_minor
            except Exception:
                logger.debug("Suppressed exception in detect_wien2k_flags()", exc_info=True)

    return flags


def estimate_kpoint_density(rkmax: float | None = None) -> dict[str, Any]:  # noqa: C901
    """
    Estimate optimal k-point density using the empirical WIEN2k heuristic:
    kpoints_per_atom ≈ 125 / (volume_per_atom)
    where volume_per_atom = unit_cell_volume / natoms.

    Reads lattice parameters from case.struct to compute unit cell volume.
    Falls back to a heuristic based on atom count if struct cannot be parsed.
    """
    result: dict[str, Any] = {
        "nkpt_est": 0,
        "kpt_per_atom": 0.0,
        "volume": 0.0,
        "formula_units": 1,
        "recommendation": "",
    }

    natoms = 1
    atom_types: dict[str, int] = {}
    volume = 0.0
    parsed_struct = False

    struct_files = list(Path(".").glob("*.struct"))
    if struct_files:
        try:
            content = struct_files[0].read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            for line in lines:
                m = re.search(r"NONEQUIV\.ATOMS\s*:\s*(\d+)", line, re.IGNORECASE)
                if m:
                    natoms = int(m.group(1))
                    break

            if natoms == 1:
                mult_matches = re.findall(r'MULT\s*=\s*(\d+)', content, re.IGNORECASE)
                if mult_matches:
                    natoms = sum(int(m) for m in mult_matches)
                else:
                    atom_lines = [
                        line for line in lines
                        if re.match(r"^\s*ATOM\s*[-\d]+:", line, re.IGNORECASE)
                    ]
                    if atom_lines:
                        natoms = len(atom_lines)

            for line in lines:
                m_type = re.match(r"^\s*ATOM\s*[-\d]+:\s*.*TOT\s*=\s*(\w+)", line, re.IGNORECASE)
                if m_type:
                    elem = m_type.group(1)
                    atom_types[elem] = atom_types.get(elem, 0) + 1

            lattice_vectors: list[list[float]] = []
            a_vector: list[float] | None = None
            b_vector: list[float] | None = None
            c_vector: list[float] | None = None
            angstrom_mode = False

            for line in lines:
                if not angstrom_mode and "ANG" in line.upper() and "LATT" not in line.upper():
                    angstrom_mode = True
                if re.match(r"^\s*[A-Za-z]+\s+VALUE", line, re.IGNORECASE):
                    angstrom_mode = True

            lat_param_lines = []
            for line in lines:
                clean = re.sub(r"#.*", "", line).strip()
                if not clean:
                    continue
                parts = clean.split()
                if len(parts) == 3 or len(parts) == 4 or len(parts) == 6:
                    try:
                        nums = [float(p) for p in parts[:3]]
                        has_enough = sum(1 for n in nums if n != 0.0) >= 1
                        is_rotation = all(n in (-1.0, 0.0, 1.0) for n in nums)
                        if has_enough and all(-200 <= n <= 200 for n in nums) and not is_rotation:
                            lat_param_lines.append(nums)
                    except (ValueError, TypeError):
                        continue

            if len(lat_param_lines) >= 3:
                a_vector = lat_param_lines[0]
                b_vector = lat_param_lines[1]
                c_vector = lat_param_lines[2]
                lattice_vectors = [a_vector, b_vector, c_vector]
            elif len(lat_param_lines) >= 1:
                for line in lines:
                    clean = re.sub(r"#.*", "", line).strip()
                    parts = clean.split()
                    if len(parts) >= 6:
                        try:
                            nums = [float(p) for p in parts]
                            a_vector = nums[:3]
                            b_vector = nums[3:6]
                            c_vector = [nums[6], nums[7], nums[8]] if len(nums) >= 9 else [1.0, 0.0, 0.0]
                            lattice_vectors = [a_vector, b_vector, c_vector]
                            break
                        except (ValueError, TypeError):
                            continue

            if lat_param_lines and not lattice_vectors:
                full = []
                for row in lat_param_lines:
                    full.extend(row)
                full = full + [0.0] * (9 - len(full))
                a_vector = full[0:3]
                b_vector = full[3:6]
                c_vector = full[6:9]
                if any(v != 0.0 for v in a_vector) and any(v != 0.0 for v in b_vector) and any(v != 0.0 for v in c_vector):
                    lattice_vectors = [a_vector, b_vector, c_vector]

            if lattice_vectors and len(lattice_vectors) == 3:
                a = lattice_vectors[0]
                b = lattice_vectors[1]
                c = lattice_vectors[2]
                cross_bc = [
                    b[1] * c[2] - b[2] * c[1],
                    b[2] * c[0] - b[0] * c[2],
                    b[0] * c[1] - b[1] * c[0],
                ]
                volume = abs(
                    a[0] * cross_bc[0] + a[1] * cross_bc[1] + a[2] * cross_bc[2]
                )
                natoms = max(natoms, 1)
                kpt_per_atom = 125.0 / (volume / natoms) if volume > 0 else 0.0
                nkpt_est = max(1, round(kpt_per_atom * natoms))

                result["nkpt_est"] = nkpt_est
                result["kpt_per_atom"] = kpt_per_atom
                result["volume"] = volume
                result["formula_units"] = natoms
                result["recommendation"] = (
                    f"Estimated {nkpt_est} k-points for {natoms} atoms "
                    f"(vol={volume:.1f} Å³, density={kpt_per_atom:.3f} kpt/atom)"
                )
                parsed_struct = True
        except Exception as e:
            logger.warning(f"Failed to parse struct for k-point density: {e}")

    if not parsed_struct:
        params = detect_problem_size()
        natoms = max(params.get("atoms", 10), 1)
        base = 8 if natoms <= 4 else (16 if natoms <= 16 else (32 if natoms <= 50 else 64))
        nkpt_est = max(1, base)
        result["nkpt_est"] = nkpt_est
        result["kpt_per_atom"] = nkpt_est / natoms
        result["volume"] = 0.0
        result["formula_units"] = natoms
        result["recommendation"] = (
            f"Heuristic estimate: {nkpt_est} k-points for {natoms} atoms "
            f"(struct file could not be parsed)"
        )

    return result


def detect_io_bottleneck(nmat: int, nkpt: int, total_cores: int) -> dict[str, Any]:
    """
    Detect potential I/O bottleneck conditions for lapw2.
    lapw2 writes large vector files; high core counts with few k-points
    can cause I/O contention on shared filesystems.
    """
    result: dict[str, Any] = {
        "warning": None,
        "auto_enable_vector_split": False,
        "suggestion": None,
        "risk_level": "low"
    }

    if nmat <= 0 or nkpt <= 0:
        return result

    if nmat > 8000 and nkpt < 4 and total_cores > 16:
        result["warning"] = (
            f"High core count ({total_cores}) with large matrix ({nmat}) and few k-points ({nkpt}) "
            "may cause I/O bottleneck in lapw2."
        )
        result["auto_enable_vector_split"] = True
        result["suggestion"] = "Auto-enabling lapw2_vector_split:4"
        result["risk_level"] = "high"
    elif nmat > 5000 and nkpt < 8 and total_cores > 32:
        result["warning"] = (
            f"Moderate I/O risk: nmat={nmat}, nkpt={nkpt}, cores={total_cores}. "
            "Monitor lapw2 performance."
        )
        result["risk_level"] = "medium"

    return result
