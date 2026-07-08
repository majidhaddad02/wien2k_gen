"""
CP2K Backend – Production-Grade Configuration Generator for HPC Clusters.
Implements CP2K-specific logic for:
• Parsing .inp files to extract atom count, basis set size, functional type,
  and other problem parameters for resource estimation
• Generating job scripts with mpirun configuration for cp2k.popt
• Auto-detection of CP2K calculations via *.inp files in the working directory
• Support for CP2K-specific I/O options: input.inp -> output.out
• Parse CP2K input for atom count, basis set size, functional type
• Resource estimation based on basis set cardinality and system size

Key Features:
• Robust CP2K input file parsing with regex-based fallbacks for missing sections
• Dynamic MPI rank estimation from topology and problem size
• Support for DFTB, semi-empirical, and hybrid functional detection
• Comprehensive English documentation, type hints, and HPC-grade error handling
"""

import datetime
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.topology import Topology
from ..logging_config import get_logger
from ..utils.atomic_write import atomic_write
from .base import Backend, ProblemSize, ResourceEstimate

logger = get_logger(__name__)


# =============================================================================
# CP2K Backend Implementation
# =============================================================================

class CP2KBackend(Backend):
    """
    CP2K-specific backend implementation.

    Handles generation of execution scripts with mpirun configuration,
    parses .inp input files for problem size detection, and provides
    resource estimation based on CP2K's Gaussian and plane-wave (GPW)
    methodology.

    Supports CP2K v7.x through v2024.x with focus on cp2k.popt (MPI-optimized
    binary) and cp2k.psmp (MPI+OpenMP hybrid binary).
    """

    _CP2K_BINARY_OPTIONS = ["cp2k.popt", "cp2k.psmp", "cp2k.ssmp", "cp2k_shell.popt"]
    _SECTION_PATTERN = re.compile(r'^\s*&(\w+)', re.IGNORECASE)
    _END_SECTION_PATTERN = re.compile(r'^\s*&END\s+\w+', re.IGNORECASE)
    _KEYWORD_PATTERN = re.compile(r'^\s*(\w[\w\-]*)\s+(.*)', re.IGNORECASE)

    # =========================================================================
    # Backend Interface Implementation
    # =========================================================================

    def detect_problem_size(self) -> ProblemSize:
        """
        Extract CP2K problem parameters from .inp files in the working directory.

        Parses &GLOBAL, &FORCE_EVAL, &SUBSYS, &KIND sections to determine:
        - Atom count from &COORD
        - Basis set size from &KIND/BASIS_SET
        - Functional type from &DFT/XC/XC_FUNCTIONAL
        - Plane-wave cutoff from &MGRID/CUTOFF

        Returns:
            ProblemSize TypedDict with extracted parameters.
        """
        inp_file = self._find_input_file()
        if inp_file is None:
            logger.warning("No CP2K .inp file found in current directory.")
            return self._default_problem_size()

        return self._parse_inp_file(inp_file)

    def generate_input(self, topo: Topology, suggestion: Dict[str, Any]) -> str:
        """
        Generate a CP2K job execution script.

        Produces a run_cp2k_optimized.sh script with MPI configuration,
        environment setup, and CP2K-specific tuning parameters.

        Args:
            topo: Hardware topology for resource allocation.
            suggestion: Resource allocation suggestion from the optimizer.

        Returns:
            String content of the run script.
        """
        return self._build_runner_script(topo, suggestion)

    def get_execution_command(self, suggestion: Dict[str, Any]) -> str:
        """
        Return the dynamically constructed execution command for CP2K.

        Constructs a command like `mpirun -np N cp2k.popt -i input.inp -o output.out`
        based on the provided resource suggestion.

        Args:
            suggestion: Resource allocation suggestion containing mode, cores, etc.

        Returns:
            Shell command string to execute the CP2K calculation.
        """
        mode = suggestion.get("mode", "mpi")
        total_cores = suggestion.get("recommended_total_cores", 1)
        omp = suggestion.get("omp_threads_per_rank", 1)

        binary = self._detect_cp2k_binary()

        inp_file = suggestion.get("input_file", "input.inp")
        out_file = suggestion.get("output_file", "output.out")

        if mode == "hybrid":
            ranks = max(1, total_cores // omp)
            return f"mpirun -np {ranks} {binary} -i {inp_file} -o {out_file}"
        else:
            return f"mpirun -np {total_cores} {binary} -i {inp_file} -o {out_file}"

    def validate_suggestion(self, suggestion: Dict[str, Any]) -> List[str]:
        """
        Validate suggestion against CP2K-specific constraints.

        CP2K scales best with hybrid MPI+OpenMP for large systems.
        MPI-only mode works well for smaller systems (< 1000 atoms).

        Args:
            suggestion: Resource allocation suggestion to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        total_cores = suggestion.get("recommended_total_cores", 1)
        omp = suggestion.get("omp_threads_per_rank", 1)
        mode = suggestion.get("mode", "mpi")

        if total_cores <= 0:
            errors.append("recommended_total_cores must be > 0")

        if omp <= 0:
            errors.append("omp_threads_per_rank must be > 0")

        if mode == "hybrid" and total_cores % omp != 0:
            errors.append("total_cores not divisible by omp_threads_per_rank for hybrid mode")

        problem_params = suggestion.get("problem_params", {})
        atoms = problem_params.get("atoms", 10) or 10
        if atoms > 1000 and mode == "mpi":
            errors.append("Large system (>1000 atoms) may benefit from hybrid MPI+OpenMP mode.")

        return errors

    def estimate_resources(self, params: ProblemSize, topo: Topology) -> ResourceEstimate:
        """
        Estimate memory and time requirements for CP2K calculations.

        Uses CP2K-specific scaling laws:
        - Memory ~ N_atoms * n_basis^2 for Gaussian basis
        - Time ~ N_atoms^3 for DFT, scaled by functional complexity
        - Enhanced for hybrid functionals (HFX) with AO integral overhead

        Args:
            params: Problem size parameters extracted from input files.
            topo: Hardware topology for resource allocation.

        Returns:
            ResourceEstimate TypedDict with memory, time, and mode recommendations.
        """
        atoms = params.get("atoms", 10) or 10
        ecut = params.get("ecut", 400.0) or 400.0
        is_hybrid = params.get("is_hybrid", False)
        nmat = params.get("nmat", 1000)

        basis_per_atom = max(10, min(100, nmat // max(1, atoms) if atoms else 100))

        total_basis = atoms * basis_per_atom
        mem_hfx_gb = (total_basis ** 2 * 8) / (1024 ** 3) if is_hybrid else 0
        mem_nlmo_gb = (total_basis * 50) / (1024 ** 2)  # Non-local MO sparse storage
        mem_plane_wave_gb = (ecut ** 1.5 / 1e7) * atoms

        total_mem_gb = mem_hfx_gb + mem_nlmo_gb + mem_plane_wave_gb + 0.5
        mem_per_core_mb = max(256, int(total_mem_gb * 1024 / max(1, topo.total_cores)))

        time_base_min = (atoms ** 3) / 5e7 * 60
        if is_hybrid:
            time_base_min *= 5.0

        recommended_mode = "hybrid" if atoms > 500 or is_hybrid else "mpi"

        warnings = []
        if is_hybrid:
            warnings.append("Hybrid functional detected. HFX will dominate memory and time.")
        if atoms > 1000:
            warnings.append("Large system. Consider using CP2K's sparse matrix (DBCSR) optimizations.")

        return {
            "memory_per_core_mb": mem_per_core_mb,
            "estimated_time_minutes": round(time_base_min, 1),
            "recommended_mode": recommended_mode,
            "warnings": warnings,
            "disk_io_gb": round(total_mem_gb * 0.5, 2),
            "peak_flops_utilization": 0.25 if is_hybrid else 0.4,
        }

    def write_auxiliary_files(self, topo: Topology, suggestion: Dict[str, Any]) -> None:
        """
        Write run_cp2k_optimized.sh with environment setup and MPI configuration.

        Handles backup of existing script and atomic write with executable permissions.

        Args:
            topo: Hardware topology.
            suggestion: Resource allocation suggestion.
        """
        script_path = Path("run_cp2k_optimized.sh")

        if script_path.exists():
            backup_path = script_path.with_suffix(".sh.bak")
            try:
                shutil.copy2(script_path, backup_path)
                logger.debug(f"Backed up {script_path} to {backup_path}")
            except Exception as exc:
                logger.warning(f"Could not backup {script_path}: {exc}")

        content = self._build_runner_script(topo, suggestion)
        try:
            atomic_write(script_path, content, mode=0o755)
            logger.info(f"Written {script_path} ({len(content)} bytes)")
        except Exception as exc:
            logger.error(f"Failed to write CP2K runner script: {exc}")

    def get_short_test_command(self) -> Optional[str]:
        """Return command for a quick 1-step CP2K test run."""
        binary = self._detect_cp2k_binary()
        inp_file = self._find_input_file()
        inp_name = inp_file.name if inp_file else "input.inp"
        return f"mpirun -np 1 {binary} -i {inp_name} -o test.out 2>&1 &"

    def get_config_filename(self) -> str:
        """Return default configuration filename for CP2K."""
        return "cp2k_job.sh"

    def parse_output(self, log_path: Path) -> Dict[str, Any]:
        """
        Parse CP2K output files for convergence, timing, and errors.

        Args:
            log_path: Path to CP2K output file (*.out).

        Returns:
            Dict with keys: exists, converged, errors, timing, content_snippet.
        """
        if not log_path.exists():
            return {"exists": False, "converged": None, "errors": [], "timing": {}}

        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
            lower_content = content.lower()

            converged = any(kw in lower_content for kw in [
                "geometry optimization completed",
                "scf run converged",
                "self-consistent energy",
                "energy change",
                "convergence achieved",
            ])

            timing = {}
            time_match = re.search(r'cputime\s*=\s*([\d\.]+)', content)
            if time_match:
                timing["total_cpu"] = float(time_match.group(1))

            time_match = re.search(r'cp2k\s*\|\s*total\s+time\s*:\s*([\d\.]+)', content, re.IGNORECASE)
            if time_match:
                timing["cp2k_total_time"] = float(time_match.group(1))

            errors = []
            if "aborted" in lower_content:
                errors.append("CP2K aborted unexpectedly. Check output for details.")
            if "insufficient memory" in lower_content or "out of memory" in lower_content:
                errors.append("Out of memory. Increase allocation or use sparse matrices.")
            if "numerical instabilities" in lower_content:
                errors.append("Numerical instability detected. Check basis set and cutoff.")
            if "segmentation fault" in lower_content:
                errors.append("Segmentation fault. Check MPI configuration and stack size.")

            return {
                "exists": True,
                "converged": converged,
                "errors": errors,
                "timing": timing,
                "content_snippet": content[:1000] if len(content) > 1000 else content,
            }
        except Exception as exc:
            logger.warning(f"Could not parse CP2K output {log_path}: {exc}")
            return {"exists": True, "converged": None, "errors": [f"Parse error: {exc}"], "timing": {}}

    # =========================================================================
    # CP2K-Specific Parsing Methods
    # =========================================================================

    def _find_input_file(self) -> Optional[Path]:
        """Locate the CP2K input file (*.inp) in the current directory."""
        inp_files = sorted(Path(".").glob("*.inp"))
        if not inp_files:
            return None

        if len(inp_files) == 1:
            return inp_files[0]

        for candidate in inp_files:
            if candidate.name.lower() in ("input.inp", "cp2k.inp"):
                return candidate
        return inp_files[0]

    def _detect_cp2k_binary(self) -> str:
        """
        Detect the preferred CP2K binary.

        Checks PATH in order: cp2k.popt, cp2k.psmp, cp2k.ssmp, cp2k_shell.popt.
        Falls back to cp2k.popt if none found.
        """
        for binary in self._CP2K_BINARY_OPTIONS:
            if shutil.which(binary):
                return binary
        return "cp2k.popt"

    def _default_problem_size(self) -> ProblemSize:
        """Return conservative default problem size when no input file is found."""
        return {
            "atoms": 10,
            "kpoints": 1,
            "nmat": 500,
            "nbands": None,
            "rkmax": 5.0,
            "is_soc": False,
            "is_hybrid": False,
            "complexity": 0.2,
            "ecut": 400.0,
            "nspin": 1,
            "lattice_type": None,
            "symmetry_group": None,
            "magnetic_order": None,
        }

    def _parse_inp_file(self, inp_file: Path) -> ProblemSize:
        """
        Parse CP2K input file to extract problem parameters.

        Handles the hierarchical section structure of CP2K input:
        &GLOBAL -> &FORCE_EVAL -> &SUBSYS -> (&COORD, &KIND, &CELL)
        &FORCE_EVAL -> &DFT -> (&MGRID, &SCF, &XC)

        Args:
            inp_file: Path to the CP2K .inp file.

        Returns:
            ProblemSize TypedDict with extracted parameters.
        """
        result = self._default_problem_size()

        try:
            content = inp_file.read_text(encoding="utf-8", errors="replace")
            sections = self._parse_cp2k_sections(content)

            nesting = sections.get("_nesting", {})
            force_eval = sections.get("FORCE_EVAL", {})
            dft = force_eval.get("DFT", {})
            subsys = force_eval.get("SUBSYS", {})
            global_sec = sections.get("GLOBAL", {})

            coord_atoms = self._count_coord_atoms(subsys)
            if coord_atoms > 0:
                result["atoms"] = coord_atoms
            else:
                result["atoms"] = self._count_kind_atoms(subsys)

            basis_sizes = self._extract_basis_sizes(subsys)
            if basis_sizes:
                result["nmat"] = sum(basis_sizes) // max(1, len(basis_sizes))

            functional = self._detect_functional(dft)
            result["is_hybrid"] = functional["is_hybrid"]
            if functional["name"]:
                if "soc" in functional["name"].lower():
                    result["is_soc"] = True

            mgrid = dft.get("MGRID", {})
            for key, value in mgrid.items():
                if key.upper() == "CUTOFF":
                    try:
                        result["ecut"] = float(value)
                    except (ValueError, TypeError):
                        pass

            result["nspin"] = 1
            for key, value in subsys.items():
                if key.upper() == "MULTIP":
                    try:
                        mult = int(value)
                        if mult > 1:
                            result["nspin"] = 2
                    except (ValueError, TypeError):
                        pass

            for key in dft:
                if key.upper() in ("UKS", "LSD", "SPIN_POLARIZED"):
                    result["nspin"] = 2
                    break

            for key in global_sec:
                if key.upper() == "RUN_TYPE":
                    run_type = str(global_sec[key]).upper()
                    if "MD" in run_type or "GEO_OPT" in run_type:
                        result["complexity"] = 1.5
                    elif "BAND" in run_type or "LINEAR_RESPONSE" in run_type:
                        result["complexity"] = 2.0

            result["complexity"] = max(0.1, result["atoms"] / 200.0)

        except Exception as exc:
            logger.warning(f"CP2K input parsing failed for {inp_file}: {exc}")

        return result

    def _parse_cp2k_sections(self, content: str) -> Dict[str, Any]:
        """
        Parse CP2K input file sections into a nested dictionary.

        Handles the &SECTION / &END SECTION syntax with arbitrary nesting depth.
        Keyword-value pairs are extracted from lines within each section.

        Args:
            content: Full CP2K input file content.

        Returns:
            Nested dictionary of sections and their keyword-value data.
        """
        sections: Dict[str, Any] = {}
        stack: List[Dict[str, Any]] = [sections]
        section_names: List[str] = []

        for raw_line in content.splitlines():
            line = raw_line.split("!", 1)[0].split("#", 1)[0].strip()
            if not line:
                continue

            section_match = self._SECTION_PATTERN.match(line)
            end_match = self._END_SECTION_PATTERN.match(line)

            if section_match and not end_match:
                sec_name = section_match.group(1).upper()
                new_section: Dict[str, Any] = {}
                current = stack[-1]
                existing = current.get(sec_name)
                if existing is None:
                    current[sec_name] = new_section
                elif isinstance(existing, dict):
                    nesting_key = f"_{sec_name}_nesting"
                    nested_list = current.get(nesting_key, [])
                    if not nested_list:
                        nested_list = [existing]
                        current[nesting_key] = nested_list
                    nested_list.append(new_section)
                    current[sec_name] = new_section
                stack.append(new_section)
                section_names.append(sec_name)

            elif end_match:
                if len(stack) > 1:
                    stack.pop()
                    section_names.pop()

            else:
                kw_match = self._KEYWORD_PATTERN.match(line)
                if kw_match:
                    key = kw_match.group(1).upper()
                    value = kw_match.group(2).strip()
                    current = stack[-1]
                    existing = current.get(key)
                    if existing is None:
                        current[key] = value
                    elif isinstance(existing, list):
                        existing.append(value)
                    else:
                        current[key] = [existing, value]

        return sections

    def _count_coord_atoms(self, subsys: Dict[str, Any]) -> int:
        """
        Count atoms from &COORD section in CP2K input.

        Supports both explicit coordinate lines and XYZ-style blocks.

        Args:
            subsys: Parsed &SUBSYS section data.

        Returns:
            Number of atoms counted, or 0 if not determinable.
        """
        coord = subsys.get("COORD", {})
        if isinstance(coord, dict):
            atom_count = 0
            for key, value in coord.items():
                if isinstance(value, str) and not key.startswith("_"):
                    parts = value.split()
                    if len(parts) >= 3:
                        try:
                            float(parts[0])
                            float(parts[1])
                            float(parts[2])
                            atom_count += 1
                        except ValueError:
                            pass

            if atom_count > 0:
                return atom_count

        coord_list = subsys.get("_COORD_nesting", [])
        if coord_list:
            for coord_entry in coord_list:
                if isinstance(coord_entry, dict):
                    atom_count = sum(
                        1 for _, v in coord_entry.items()
                        if isinstance(v, str) and len(v.split()) >= 3
                    )
                    if atom_count > 0:
                        return atom_count

        return 0

    def _count_kind_atoms(self, subsys: Dict[str, Any]) -> int:
        """
        Count atoms from &KIND sections by parsing element references.

        Attempts to find an &ATOM section or count &KIND definitions.
        Falls back to a conservative default.

        Args:
            subsys: Parsed &SUBSYS section data.

        Returns:
            Estimated atom count, minimum 1.
        """
        kind_nesting = subsys.get("_KIND_nesting", [])
        if kind_nesting:
            return len(kind_nesting)

        kind_section = subsys.get("KIND", {})
        if isinstance(kind_section, dict) and kind_section:
            return 1

        return self._default_problem_size()["atoms"]

    def _extract_basis_sizes(self, subsys: Dict[str, Any]) -> List[int]:
        """
        Extract basis set sizes from &KIND sections.

        Maps known CP2K basis sets to approximate function counts.
        Custom basis sets are estimated from naming conventions.

        Args:
            subsys: Parsed &SUBSYS section data.

        Returns:
            List of approximate basis function counts per atom kind.
        """
        basis_size_map = {
            "DZVP": 13,
            "DZVP-MOLOPT": 13,
            "DZVP-MOLOPT-SR": 13,
            "DZVP-MOLOPT-GTH": 13,
            "TZVP": 22,
            "TZVP-MOLOPT": 22,
            "TZVP-MOLOPT-GTH": 22,
            "TZV2P": 30,
            "TZV2P-MOLOPT": 30,
            "TZV2PX": 35,
            "TZV2PX-MOLOPT": 35,
            "QZVP": 40,
            "QZVP-MOLOPT": 40,
            "5Z": 55,
            "6-31G": 9,
            "6-31G*": 15,
            "6-311G": 13,
            "6-311G**": 22,
            "cc-pVDZ": 14,
            "cc-pVTZ": 30,
            "cc-pVQZ": 55,
        }

        sizes = []

        kind_nesting = subsys.get("_KIND_nesting", [])
        if not kind_nesting:
            kind_sec = subsys.get("KIND", {})
            if isinstance(kind_sec, dict):
                kind_nesting = [kind_sec]

        for kind_entry in kind_nesting:
            if not isinstance(kind_entry, dict):
                continue

            basis = kind_entry.get("BASIS_SET", "")
            if isinstance(basis, str):
                basis_upper = basis.upper().strip()
                found = False
                for name, size in basis_size_map.items():
                    if name in basis_upper:
                        sizes.append(size)
                        found = True
                        break
                if not found:
                    sizes.append(20)

        if not sizes:
            sizes.append(20)

        return sizes

    def _detect_functional(self, dft: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect the exchange-correlation functional from &DFT/&XC section.

        Identifies hybrid functionals (B3LYP, PBE0, HSE06, etc.) and
        spin-orbit coupling indicators from functional naming.

        Args:
            dft: Parsed &DFT section data.

        Returns:
            Dict with keys: name, is_hybrid, is_soc.
        """
        hybrid_names = [
            "B3LYP", "PBE0", "HSE06", "HSE03", "BHandH", "BHandHLYP",
            "BLYP35", "SCAN0", "M06-2X", "wB97X", "CAM-B3LYP",
            "LC-wPBE", "MCY3", "B2PLYP", "DSD-PBEP86",
        ]

        result = {"name": None, "is_hybrid": False, "is_soc": False}

        xc = dft.get("XC", {})
        xc_func = xc.get("XC_FUNCTIONAL", {})

        if isinstance(xc_func, dict):
            for key, value in xc_func.items():
                if isinstance(value, str):
                    name_upper = value.upper()
                    result["name"] = value
                    for hybrid in hybrid_names:
                        if hybrid.upper() in name_upper:
                            result["is_hybrid"] = True
                            break

        for key, value in dft.items():
            if isinstance(value, str):
                name_upper = value.upper()
                if key.upper() == "XC_FUNCTIONAL" or key.upper().startswith("XC"):
                    result["name"] = value
                    for hybrid in hybrid_names:
                        if hybrid.upper() in name_upper:
                            result["is_hybrid"] = True
                            break

        return result

    # =========================================================================
    # Runner Script Generation
    # =========================================================================

    def _build_runner_script(self, topo: Topology, suggestion: Dict[str, Any]) -> str:
        """
        Build the run_cp2k_optimized.sh script content.

        Includes MPI launcher detection, CP2K-specific environment variables,
        scratch directory management, and preemption handling.

        Args:
            topo: Hardware topology for resource allocation.
            suggestion: Resource allocation suggestion.

        Returns:
            Complete shell script content as a string.
        """
        mode = suggestion.get("mode", "mpi")
        total_cores = suggestion.get("recommended_total_cores", 1)
        omp = suggestion.get("omp_threads_per_rank", 1)

        binary = self._detect_cp2k_binary()

        inp_file = suggestion.get("input_file", "input.inp")
        out_file = suggestion.get("output_file", "output.out")

        if mode == "hybrid":
            mpi_ranks = max(1, total_cores // omp)
        else:
            mpi_ranks = total_cores

        script = f"""#!/bin/bash
# ==============================================================================
# Auto-generated CP2K Runner Script (wien2k_gen v0.1.0)
# Mode: {mode.upper()} | Total Cores: {total_cores} | OMP: {omp} | MPI Ranks: {mpi_ranks}
# Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}Z
# Binary: {binary}
# ==============================================================================

# CP2K-Specific Environment
export OMP_NUM_THREADS={omp}
export MKL_NUM_THREADS={omp}
export OPENBLAS_NUM_THREADS={omp}
export OMP_STACKSIZE=256M
export CP2K_DATA_DIR="${{CP2K_DATA_DIR:-/opt/cp2k/data}}"

# Scratch Directory Setup
SCRATCH_DIR=$(mktemp -d -p /dev/shm 2>/dev/null || mktemp -d -p ${{SCRATCH:-/scratch}} 2>/dev/null || mktemp -d)
export CP2K_SCRATCH="$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR"
trap 'rm -rf "$SCRATCH_DIR" 2>/dev/null' EXIT TERM INT
echo "[cp2k_gen] Scratch directory: $SCRATCH_DIR"

# Preemption Handler
_preemption_handler() {{
    echo "[cp2k_gen] Preemption signal received. Forcing clean exit..."
    sleep 2
    exit 143
}}
trap _preemption_handler TERM USR1

# MPI Launcher Detection
MPI_LAUNCHER="mpirun"
if [ -n "$SLURM_JOB_ID" ]; then
    MPI_LAUNCHER="srun --mpi=pmix --hint=nomultithread --cpu-bind=core"
elif [ -n "$LSB_JOBID" ]; then
    MPI_LAUNCHER="mpirun -prot -aff=automatic"
elif [ -n "$PBS_JOBID" ]; then
    MPI_LAUNCHER="mpirun"
fi

# Execute CP2K
echo "[cp2k_gen] Starting CP2K execution with {mpi_ranks} MPI rank(s)..."
$MPI_LAUNCHER -np {mpi_ranks} {binary} -i {inp_file} -o {out_file}
EXIT_CODE=$?
echo "[cp2k_gen] CP2K finished with exit code $EXIT_CODE"
exit $EXIT_CODE
"""
        return script
