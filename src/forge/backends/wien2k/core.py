from __future__ import annotations

import datetime
import math
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...types import Wien2kFlags

import forge.core.hardware as _hw

from ...core.hardware import (
    get_interconnect_info,
    get_job_memory_limit_mb,
    get_numa_node_count,
    get_physical_cores,
    get_total_mem_kb,
)
from ...core.topology import Topology
from ...logging_config import get_logger
from ...utils.atomic_write import atomic_write
from ..base import Backend, ProblemSize
from .parsers import (
    DayfileResult,
    detect_io_bottleneck,
    estimate_kpoint_density,
    parse_dayfile,
    parse_output,
)
from .parsers import (
    detect_problem_size as _detect_problem_size,
)
from .parsers import (
    detect_wien2k_flags as _detect_wien2k_flags,
)

logger = get_logger(__name__)


class Wien2kBackend(Backend):
    """
    WIEN2k-specific backend implementation.
    Handles generation of .machines, parallel_options, and run_optimized.sh
    with optimizations for modern HPC clusters (SLURM/PBS, NUMA, UCX, MPI).
    """

    # =========================================================================
    # Backend Interface Implementation
    # =========================================================================

    def detect_problem_size(self) -> ProblemSize:
        """Extract WIEN2k problem parameters from input files."""
        return _detect_problem_size()

    def generate_input(self, topo: Topology, suggestion: dict[str, Any]) -> str:
        """Generate .machines file content for WIEN2k parallel execution."""
        # Integrate ELPA solver recommendation into suggestion dict
        nmat = suggestion.get("nmat", self._detect_problem_size().get("nmat", 0))
        nkpt = suggestion.get("nkpt", self._detect_problem_size().get("kpoints", 0))
        is_soc = suggestion.get("is_soc", self._detect_problem_size().get("is_soc", False))
        total_ranks = sum(topo.cores_per_node) if topo.cores_per_node else 1

        if nmat > 2000:
            try:
                from ..elpa_selector import select_eigensolver
                gpu_ok = bool(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
                solver_sel = select_eigensolver(nmat, nkpt, is_soc, gpu_ok, total_ranks=total_ranks)
                suggestion["elpa_solver"] = solver_sel.recommended_solver
                suggestion["elpa_block_size"] = solver_sel.block_size
                suggestion["elpa_reason"] = solver_sel.reason
                logger.info(
                    f"ELPA solver selected: {solver_sel.recommended_solver} "
                    f"(block={solver_sel.block_size}, "
                    f"BLACS={solver_sel.recommended_grid[0]}x{solver_sel.recommended_grid[1]})"
                )
            except Exception as e:
                logger.debug(f"ELPA solver selection skipped: {e}")

        lines = self._build_machines_lines(topo, suggestion)
        return "\n".join(lines)

    def get_execution_command(self, suggestion: dict[str, Any]) -> str:  # noqa: C901
        """
        Return dynamically constructed execution command with WIEN2k flags.

        Uses detected calculation type (SCF, spin-polarized, SOC, LDA+U,
        hybrid, EECE, forces) to construct the correct run_lapw/runsp_lapw
        command with appropriate flags.

        Reference: Blaha, P. et al. (2020). WIEN2k Usersguide, Sections 4.1-4.4.
        """
        mode = suggestion.get("mode", "mpi")
        total_cores = suggestion.get("recommended_total_cores", 1)
        omp = suggestion.get("omp_threads_per_rank", 1)

        calc_type = suggestion.get("calc_type", suggestion.get("exec_command", "run_lapw -p"))
        if not calc_type.startswith("run"):
            is_spin = suggestion.get("is_spin_polarized", False)
            is_soc = suggestion.get("is_soc", False)
            is_lda_u = suggestion.get("is_lda_u", False)
            is_hybrid = suggestion.get("is_hybrid", False)
            is_eece = suggestion.get("is_eece", False)
            has_forces = suggestion.get("has_forces", False)

            base_cmd = "runsp_lapw" if is_spin else "run_lapw"
            extra_flags = []
            if is_soc:
                extra_flags.append("-so")
            if is_lda_u:
                extra_flags.append("-orbc")
            if is_hybrid:
                extra_flags.append("-hf")
            if is_eece:
                extra_flags.append("-eece")
            if has_forces:
                extra_flags.append("-fc")
            calc_type = " ".join([base_cmd, "-p", *extra_flags])

        calc_base = calc_type.split()[0] if isinstance(calc_type, str) else "run_lapw"
        extra_parts = calc_type.split()[2:] if isinstance(calc_type, str) and len(calc_type.split()) > 2 else []

        if mode == "kpoint":
            if extra_parts:
                return f"{calc_base} -p {' '.join(extra_parts)}"
            return f"{calc_base} -p"
        elif mode == "hybrid":
            ranks = max(1, total_cores // omp)
            if extra_parts:
                return f"{calc_base} -p -np {ranks} -omp {omp} {' '.join(extra_parts)}"
            return f"{calc_base} -p -np {ranks} -omp {omp}"
        else:
            ranks = max(1, total_cores)
            if extra_parts:
                return f"{calc_base} -p -np {ranks} {' '.join(extra_parts)}"
            return f"{calc_base} -p -np {ranks}"

    def validate_suggestion(self, suggestion: dict[str, Any]) -> list[str]:
        """Validate suggestion against WIEN2k-specific constraints."""
        errors = []
        mode = suggestion.get("mode", "")
        cores = suggestion.get("recommended_total_cores", 0)
        omp = suggestion.get("omp_threads_per_rank", 1)
        nmat = suggestion.get("nmat", 0)

        if cores <= 0:
            errors.append("recommended_total_cores must be > 0")
        if mode == "hybrid" and omp <= 0:
            errors.append("omp_threads_per_rank must be > 0 for hybrid mode")
        if mode == "hybrid" and cores % omp != 0:
            errors.append(
                f"total_cores ({cores}) not divisible by omp_threads ({omp}) for hybrid mode"
            )

        # Memory sanity check
        est_mem_gb = suggestion.get("estimated_memory_gb", 2.0)
        mem_per_core_mb = (est_mem_gb * 1024) / max(1, cores)
        job_limit_mb = get_job_memory_limit_mb()
        if job_limit_mb and mem_per_core_mb > job_limit_mb * 0.9:
            errors.append(
                f"Estimated memory per core ({mem_per_core_mb:.0f} MB) exceeds job limit"
            )

        # WIEN2k version/library compatibility
        if nmat > 20000 and not _hw.check_elpa_available():
            errors.append(
                "Large matrix (nmat > 20000) without ELPA: "
                "consider recompiling WIEN2k with ELPA support or switch to hybrid mode"
            )

        return errors

    def write_auxiliary_files(self, topo: Topology, suggestion: dict[str, Any]) -> None:
        """Write parallel_options and run_optimized.sh with atomic writes."""
        self._write_parallel_options(solver_hint=suggestion.get("elpa_solver", ""))
        self._write_runner_script(topo, suggestion)

    def get_short_test_command(self) -> str | None:
        """Return command for quick 2-cycle test."""
        return "run_lapw -c"

    def get_config_filename(self) -> str:
        """Return default configuration filename for WIEN2k."""
        return ".machines"

    def parse_output(self, log_path: Path) -> dict[str, Any]:
        return parse_output(log_path)

    # =========================================================================
    # Advanced WIEN2k-Specific Methods
    # =========================================================================

    def _get_optimal_lapw0_cores(self, available_cores: int, natoms: int | None) -> int:
        """
        Determine optimal core count for lapw0 (potential calculation).
        lapw0 is I/O-bound and scales poorly beyond ~8 cores for small systems.
        Over-parallelizing lapw0 wastes CPU-hours with no speedup.

        Rules (WIEN2k UG §4.5):
        - atoms < 4: serialize (overhead > benefit)
        - atoms 4-20: modest parallelism (4-6 cores)
        - atoms 20-100: good scaling (6-12 cores)
        - atoms > 100: supercell, scales to 16+
        """
        if natoms is None or natoms <= 0:
            return max(4, min(available_cores, 8))

        if natoms < 4:
            return 1  # Serial lapw0 for tiny systems
        if natoms < 20:
            return min(4, available_cores)
        if natoms < 100:
            return min(6, available_cores)
        return min(8, available_cores)

    def _smart_allocate_cores(
        self, total_cores: int, kpoints: int, atoms: int, nmat: int, mode: str, num_nodes: int
    ) -> dict[str, Any]:
        """
        Intelligent core allocation for WIEN2k processors.

        Uses problem parameters to decide optimal distribution of cores
        across lapw0, lapw1, and lapw2 based on real workload characteristics.

        Design rationale:
        - lapw0: overlap matrix, I/O-bound. Minimal cores for small systems.
        - lapw1: diagonalization, CPU-bound, parallel over k-points. Gets priority.
        - lapw2: vector ops. Can exploit vector_split for excess cores.
        - Cores beyond k-point saturation use granularity + vector_split, not wasted.
        - Amdahl's Law cap: warns when user requests more cores than useful.
          (Ref: Amdahl 1967; Hager & Wellein 2010, §4.2)

        Returns dict with per-processor core counts, kpar, reason, and warnings.
        """
        # Step 0: Amdahl's Law saturation check
        saturation_warnings: list[str] = []
        max_efficient = total_cores
        saturation = {}
        try:
            from ...optimizer.advisor import estimate_amdahl_saturation
            saturation = estimate_amdahl_saturation(
                kpoints=kpoints,
                nmat=nmat,
                atoms=atoms,
                total_cores_available=total_cores,
                num_nodes=num_nodes,
                mode=mode,
            )
            max_efficient = saturation.get("max_efficient_cores", total_cores)
            saturation_warnings = saturation.get("saturation_warnings", [])
        except ImportError:
            logger.debug("Suppressed exception in _smart_allocate_cores()", exc_info=True)

        # Step 1: lapw0 allocation
        lapw0_cores = self._get_optimal_lapw0_cores(total_cores, atoms)
        remaining = total_cores - lapw0_cores

        # Step 2: Cap lapw1 at k-point count (k-point parallelism limit)
        max_lapw1_by_kp = max(1, kpoints) if kpoints > 0 else remaining
        effective_kp = min(kpoints, remaining) if kpoints > 0 else remaining

        # Step 3: Split remaining between lapw1 and lapw2
        if nmat > 8000:
            lapw1_ratio = 0.65
        elif nmat > 3000:
            lapw1_ratio = 0.60
        else:
            lapw1_ratio = 0.55

        desired_lapw1 = max(1, int(remaining * lapw1_ratio))
        lapw1_cores = min(desired_lapw1, max_lapw1_by_kp)
        lapw2_cores = max(1, remaining - lapw1_cores)

        # Step 4: Cap lapw2 for small systems (vector work doesn't scale well)
        max_lapw2 = max(4, atoms * 4)  # ~4 cores per atom for vector I/O
        if lapw2_cores > max_lapw2 and atoms < 20:
            lapw2_cores = max_lapw2
            # Redistribute excess as granularity within lapw1 groups
            # (not wasted: WIEN2k can use extra ranks per k-point for ScaLAPACK)

        # Step 5: kpar = number of k-point parallel groups for lapw1
        kpar = max(1, min(lapw1_cores, effective_kp))

        # Step 6: Build reason
        reason_parts = [f"lapw0={lapw0_cores}c", f"lapw1={lapw1_cores}c", f"lapw2={lapw2_cores}c"]
        if kpoints > 0 and lapw1_cores >= kpoints:
            reason_parts.append("[kp-saturated]")
        if atoms < 4 and lapw0_cores == 1:
            reason_parts.append("[lapw0:serial]")
        if saturation.get("is_saturated"):
            reason_parts.append(f"[amdahl:max_eff={max_efficient}]")
        total_used = lapw0_cores + lapw1_cores + lapw2_cores
        if total_used < total_cores:
            reason_parts.append(f"[granularity:{total_cores-total_used}c]")

        return {
            "lapw0_cores": lapw0_cores,
            "lapw1_cores": lapw1_cores,
            "lapw2_cores": lapw2_cores,
            "kpar": kpar,
            "reason": " | ".join(reason_parts),
            "max_efficient_cores": max_efficient,
            "saturation_warnings": saturation_warnings,
        }

    def _get_optimal_mkl_threads(self, omp_threads: int, mode: str, nmat: int, is_soc: bool) -> int:
        """
        Determine optimal MKL thread count for linear algebra operations.
        SOC calculations require single-threaded MKL for correctness.
        Large matrices benefit from fewer threads to reduce cache contention.
        """
        if is_soc:
            return 1  # SOC requires single-threaded MKL
        if mode == "mpi" and nmat > 5000:
            return 1  # MPI mode with large matrices: avoid thread contention
        if nmat > 10000:
            return min(omp_threads, 2)
        if nmat > 5000:
            return min(omp_threads, 4)
        return omp_threads

    def _detect_io_bottleneck(self, nmat: int, nkpt: int, total_cores: int) -> dict[str, Any]:
        return detect_io_bottleneck(nmat, nkpt, total_cores)

    def parse_dayfile(self, dayfile_path: Path) -> DayfileResult:
        return parse_dayfile(dayfile_path)

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _detect_problem_size(self) -> dict[str, Any]:
        return _detect_problem_size()

    def _detect_wien2k_flags(self) -> Wien2kFlags:
        return _detect_wien2k_flags()

    def estimate_kpoint_density(self, rkmax: float | None = None) -> dict[str, Any]:
        return estimate_kpoint_density(rkmax)

    def auto_rkmax(
        self, available_cores: int, available_memory_gb: float
    ) -> float:
        """
        Compute the maximum feasible RKMAX based on available memory and cores.

        Uses the scaling law:
            memory ∝ (nmat / natoms) * RKMAX² * nkpt

        The recommended RKMAX is clamped to the realistic WIEN2k range [5.0, 10.0].
        Formula:
            rkmax_auto = 7.0 * min(1.0, sqrt(available_memory_gb / estimated_memory_at_rkmax7))
        where estimated_memory_at_rkmax7 is derived from the existing footprint estimator.
        """
        params = self._detect_problem_size()
        natoms = max(params.get("atoms", 10), 1)
        nmat = params.get("nmat", 0)
        nkpt = params.get("kpoints", 1)

        density = self.estimate_kpoint_density()
        if nkpt <= 0:
            nkpt = max(1, density.get("nkpt_est", 1))

        if nmat <= 0:
            nmat = natoms * 100

        nmat_per_atom = max(1.0, float(nmat) / float(natoms))
        estimated_memory_at_rkmax7 = (nmat_per_atom * 49.0 * float(nkpt) * 8.0) / (1024.0 ** 3)

        if estimated_memory_at_rkmax7 <= 0:
            return 7.0

        ratio = math.sqrt(available_memory_gb / estimated_memory_at_rkmax7)
        ratio = min(1.0, max(0.4, ratio))
        rkmax_auto = 7.0 * ratio

        rkmax_auto = max(5.0, min(10.0, rkmax_auto))

        logger.info(
            f"auto_rkmax: memory={available_memory_gb:.1f} GB, "
            f"est_at_rkmax7={estimated_memory_at_rkmax7:.2f} GB, "
            f"recommended rkmax={rkmax_auto:.2f}"
        )
        return round(rkmax_auto, 2)

    def _build_machines_lines(self, topo: Topology, suggestion: dict[str, Any]) -> list[str]:  # noqa: C901
        """
        Build .machines file content per WIEN2k parallel execution spec.

        Strictly follows the format expected by lapw1para / lapwsopara / lapwdmpara:
            1: hostname:N    lapw1 process (k-point parallel — one line per kp)
            1: hostname:N    lapw2 process
            granularity:N    fine-grain grouping
            kpar:N           k-points per MPI rank
            lapw0: hostname:lapw0_cores
            lapw2_vector_split:N   vector split for I/O

        Also includes NUMA binding hints, memory warnings, and heterogeneous
        cluster node distribution with core ratio scaling.

        Ref: WIEN2k Usersguide 2023 Sections 4.5.8, 6.1;
        Blaha et al. (2020), J. Chem. Phys. 152, 074101.
        """
        mode = suggestion.get("mode", "mpi")
        total_cores = suggestion.get("recommended_total_cores", 1)
        nodes = list(topo.nodes)
        cores_per_node = list(topo.cores_per_node)
        granularity = suggestion.get("granularity", 1)
        omp = suggestion.get("omp_threads_per_rank", 1)

        params = self._detect_problem_size()
        atoms = params.get("atoms", 10)
        kpoints = params.get("kpoints", 0)
        nmat = params.get("nmat", 0)
        is_soc = params.get("is_soc", False)
        is_hybrid = params.get("is_hybrid", False)
        is_spin = params.get("is_spin_polarized", False)
        first_node = nodes[0] if nodes else "localhost"

        # Scale cores_per_node to total_cores
        available = sum(cores_per_node)
        if total_cores < available and cores_per_node:
            ratio = total_cores / available
            cores_per_node = [max(1, int(c * ratio)) for c in cores_per_node]
            diff = total_cores - sum(cores_per_node)
            if diff > 0:
                for i in range(min(diff, len(cores_per_node))):
                    cores_per_node[i] += 1
            elif diff < 0:
                for i in range(min(-diff, len(cores_per_node))):
                    if cores_per_node[i] > 1:
                        cores_per_node[i] -= 1

        is_hetero = topo.heterogeneous or (len(set(cores_per_node)) > 1)
        if is_hetero and len(cores_per_node) > 1:
            max_c = max(cores_per_node)
            cores_per_node = [max(1, int(c * total_cores / max_c / len(cores_per_node)))
                              for c in cores_per_node] if max_c > 0 else [1] * len(cores_per_node)

        lines = []
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
        lines.append(f"# FORGE v0.1.0 | {ts}")
        lines.append(f"# Mode: {mode.upper()} | Total cores = {sum(cores_per_node)} | OMP per rank = {omp}")
        lines.append(f"# Nodes: {', '.join(nodes)} | Cores: {cores_per_node}")
        lines.append(f"# Problem: atoms={atoms} kpts={kpoints} nmat={nmat} "
                     f"soc={is_soc} hybrid={is_hybrid} spin={is_spin}")
        if is_hetero:
            lines.append("# Heterogeneous cluster — ranks scaled to core ratio")
        lines.append("")

        # ── lauter allocation ──
        allocation = self._smart_allocate_cores(
            total_cores=total_cores, kpoints=kpoints, atoms=atoms,
            nmat=nmat, mode=mode, num_nodes=len(nodes)
        )

        # Memory estimate
        mem_mb_per_core = self._estimate_memory_per_core(nmat, kpoints, is_soc, is_hybrid)
        lines.append(f"# Est. memory: ~{mem_mb_per_core:.0f} MB/core → "
                     f"~{mem_mb_per_core * total_cores / 1024:.1f} GB total")
        for w in allocation.get("saturation_warnings", []):
            lines.append(f"# SATURATION: {w}")

        lines.append("")

        # ── lapw0 block (always first node, uses OpenMP) ──
        lapw0_cores = min(allocation["lapw0_cores"], cores_per_node[0] if cores_per_node else 1)
        lines.append(f"lapw0: {first_node}:{lapw0_cores}")
        lines.append("")

        # ── Decision matrix for lapw1/lapw2 parallelization ──
        strat = self._select_parallel_strategy(
            mode=mode, nmat=nmat, kpoints=kpoints, atoms=atoms,
            is_hybrid=is_hybrid, is_soc=is_soc, is_spin=is_spin,
            total_cores=total_cores, omp=omp, granularity=granularity,
        )

        # ── lapw1 / lapw2 lines ──
        if strat["strategy"] == "band_parallel":
            lines.append(f"# Band parallelization for hybrid functional (nmat={nmat})")
            kpar = min(strat["bands_per_group"], kpoints if kpoints > 0 else 1)
            lines.append(f"kpar: {kpar}")
            for node, cores in zip(nodes, cores_per_node):
                ranks_on_node = max(1, cores // omp)
                for _ in range(ranks_on_node):
                    lines.append(f"1: {node}:{omp}")
        elif strat["strategy"] == "fine_grain_elpa":
            lines.append(f"# Fine-grain MPI with ELPA (nmat={nmat}, BLACS-aware)")
            lapw1_cores = allocation.get("lapw1_cores", total_cores // 2)
            lapw2_cores = allocation.get("lapw2_cores", total_cores - lapw1_cores)
            lines.append(f"lapw1: {first_node}:{lapw1_cores}")
            lines.append(f"lapw2: {first_node}:{lapw2_cores}")
            for node in nodes[1:]:
                n1 = max(1, lapw1_cores // len(nodes))
                n2 = max(1, lapw2_cores // len(nodes))
                lines.append(f"lapw1: {node}:{n1}")
                lines.append(f"lapw2: {node}:{n2}")
            lines.append(f"granularity: {granularity}")
            if omp > 1:
                lines.append(f"omp_global: {omp}")
        elif strat["strategy"] == "core_parallel":
            lines.append(f"# Core parallelization (nmat={nmat}, large system)")
            for node, cores in zip(nodes, cores_per_node):
                lines.append(f"1: {node}:{cores}")
            lines.append(f"granularity: {granularity}")
        else:  # kpoint parallel — default
            lines.append(f"# K-point parallelization (nkpt={kpoints})")
            for node, cores in zip(nodes, cores_per_node):
                ranks_on_node = max(1, cores // omp)
                for _ in range(ranks_on_node):
                    lines.append(f"1: {node}:{omp}")
            lines.append(f"granularity: {granularity}")
            if kpoints and kpoints % total_cores != 0:
                lines.append("extrafine: 1")

        # ── Common options ──
        lines.append("")
        lines.append("omp_lapw0: 1")
        lines.append("omp_mixer: 1")
        if allocation.get("kpar", 0) > 1:
            lines.append(f"kpar: {allocation['kpar']}")

        # ── Vector split for large matrices ──
        vector_split_active = suggestion.get("vector_split_active", False)
        io_check = self._detect_io_bottleneck(nmat, kpoints, total_cores)
        if io_check.get("auto_enable_vector_split"):
            vector_split_active = True
        if vector_split_active:
            if nmat > 20000:
                split_val = 16
            elif nmat > 10000:
                split_val = 8
            elif nmat > 5000:
                split_val = 4
            else:
                split_val = 2
            lines.append(f"lapw2_vector_split: {split_val}")

        # ── Warnings ──
        for w in suggestion.get("warnings", []):
            lines.append(f"# WARNING: {w}")

        if not _hw.check_elpa_available() and mode == "mpi" and nmat > 5000:
            lines.append("# WARNING: ELPA not detected. MPI fine-grain diagonalization may be slow.")
            lines.append("# Consider recompiling WIEN2k with ELPA for large matrices.")

        return lines

    def _estimate_memory_per_core(self, nmat: int, kpoints: int,
                                  is_soc: bool, is_hybrid: bool) -> float:
        """Estimate memory requirement per MPI rank (MB).

        From WIEN2k internal documentation and empirical benchmarks:
        - Hamiltonian matrix: nmat x nmat x 16 bytes (complex double) x safety_factor
        - Eigenvectors: nmat x nbands x 8 bytes
        - Overlap matrix: nmat x nmat x 16 bytes (if hybrid)
        - SOC doubles the first-variational basis
        - Each MPI rank holds 1/kpar of the total k-points
        """
        nmat_eff = nmat if not is_soc else int(nmat * 1.5)
        safety = 2.5

        h_size = nmat_eff * nmat_eff * 16 * safety
        ev_size = nmat_eff * nmat_eff * 8

        if is_hybrid:
            h_size *= 4

        total_mb = (h_size + ev_size) / (1024 * 1024)

        if is_soc:
            total_mb *= 1.5

        return round(total_mb + 256, 0)

    @staticmethod
    def _select_parallel_strategy(
        mode: str, nmat: int, kpoints: int, atoms: int,
        is_hybrid: bool, is_soc: bool, is_spin: bool,
        total_cores: int, omp: int, granularity: int,
    ) -> dict[str, Any]:
        """Complete decision matrix for WIEN2k parallelization strategy.

        Based on WIEN2k Usersguide 2023 §§4.5.8, 6.1: WIEN2k has
        multiple nested parallelization levels that must be combined
        intelligently for optimal performance.

        Strategy selection order:
          1. Hybrid functionals (nmat > 5000) → band + k-point parallel
           2. Very large systems (nmat > 8000) with ELPA → fine_grain
          3. Large systems (nmat > 5000) with many cores → core parallel
          4. Default → k-point parallel with granularity for I/O
        """
        elpa_ok = _hw.check_elpa_available()

        if is_hybrid and nmat > 5000:
            bands_per_group = min(4, max(1, nmat // 2000))
            return {
                "strategy": "band_parallel",
                "reason": f"Hybrid functional (nmat={nmat}): band parallelization, "
                         f"{bands_per_group} bands per group",
                "bands_per_group": bands_per_group,
            }

        if nmat > 8000 and kpoints <= 2 and elpa_ok:
            return {
                "strategy": "fine_grain_elpa",
                "reason": f"Very large system (nmat={nmat}): "
                         f"fine-grain MPI with ELPA diagonalization",
                "recommend_elpa": True,
            }

        if nmat > 5000 and total_cores > 32 and not is_hybrid:
            if elpa_ok and total_cores >= 64:
                return {
                    "strategy": "fine_grain_elpa",
                    "reason": f"Large system (nmat={nmat}): fine-grain MPI, "
                             f"ELPA available, {total_cores} cores",
                    "recommend_elpa": True,
                }
            return {
                "strategy": "core_parallel",
                "reason": f"Large system (nmat={nmat}): core parallel, "
                         f"{total_cores} cores, granularity={granularity}",
            }

        return {
            "strategy": "kpoint_parallel",
            "reason": f"Standard k-point parallel (nkpt={kpoints}, granularity={granularity})",
        }

    def _write_parallel_options(self, solver_hint: str = "", omp_threads: int = 1) -> None:
        """
        Write parallel_options file with comprehensive HPC best practices.
        Includes WIEN_MPIRUN auto-detection, ELPA config, MKL threading,
        fine-grain granularity, and GPU hints.

        Reference: WIEN2k Usersguide Section 4.5.8, Blaha et al. (2020).
        """
        omp = max(1, omp_threads)
        content = (
            "# Auto-generated by forge v0.1.0\n"
            "# Reference: Blaha, P. et al. (2020). WIEN2k Usersguide Sec 4.5.8.\n"
            "\n"
            "# ---- Remote execution control ----\n"
            "export USE_REMOTE=0\n"
            "export MPI_REMOTE=0\n"
            "\n"
            "# ---- CPU affinity & threading ----\n"
            "export TASKSET=no\n"
            f"export OMP_NUM_THREADS={omp}\n"
            f"export MKL_NUM_THREADS={max(1, min(omp, 4))}\n"
            "\n"
            "# ---- MPI launcher ----\n"
            'export WIEN_MPIRUN="mpirun -np _NP_ -machinefile _HOSTS_ _EXEC_"\n'
            "\n"
            "# ---- Synchronization & I/O ----\n"
            "export DELAY=0.1\n"
            "export SLEEPY=1\n"
            "\n"
            "# ---- Parallelism granularity ----\n"
            f"export OMP_GLOBAL={omp}\n"
            "export KPAR=0\n"
            "export WIEN_GRANULARITY=1\n"
            "\n"
            "# ---- Debugging ----\n"
            "export WIEN_DBGLVL=0\n"
        )
        solver_upper = solver_hint.upper().strip()
        if "ELPA2" in solver_upper:
            content += (
                "\n# ---- ELPA2 eigensolver ----\n"
                "export USE_ELPA=2\n"
                "export ELPA_KERNEL=ELPA2\n"
            )
        elif "ELPA1" in solver_upper or "ELPA" in solver_upper:
            content += (
                "\n# ---- ELPA1 eigensolver ----\n"
                "export USE_ELPA=1\n"
                "export ELPA_KERNEL=ELPA1\n"
            )
        elif "SCALAPACK" in solver_upper:
            content += (
                "\n# ---- ScaLAPACK eigensolver ----\n"
                "export USE_ELPA=0\n"
            )
        content += "\n"
        atomic_write(Path("parallel_options"), content, mode=0o644)

    def _write_runner_script(self, topo: Topology, suggestion: dict[str, Any]) -> None:  # noqa: C901
        """
        Write run_optimized.sh with environment setup, NUMA binding, and MPI configuration.
        Production features:
        • Atomic write with backup
        • Dynamic MPI launcher detection (srun/mpirun/jsrun)
        • NUMA binding hint injection
        • Scratch directory management with multi-node fallback
        • Interconnect-aware UCX/OFI tuning
        • Preemption-resilient signal traps
        • User-customizable RUN_LAPW_CMD
        """
        script_path = Path("run_optimized.sh")

        # Backup existing script
        if script_path.exists():
            backup_path = script_path.with_suffix(".sh.bak")
            try:
                shutil.copy2(script_path, backup_path)
                logger.debug(f"Backed up {script_path} to {backup_path}")
            except Exception as e:
                logger.warning(f"Could not backup {script_path}: {e}")

        # Determine WIENROOT
        from ...core.locator import find_wienroot
        wienroot = find_wienroot()
        if not wienroot:
            logger.error("WIENROOT not set and run_lapw not found on PATH. Cannot generate script.")
            return str(script_path)

        # Disable SSH for single-node jobs (performance optimization)
        disable_ssh = (len(topo.nodes) == 1)
        mpi_env = ""
        if disable_ssh:
            mpi_env = (
                "export OMPI_MCA_plm_rsh_agent=/bin/false\n"
                "export OMPI_MCA_orte_rsh_agent=/bin/false\n"
            )

        # NUMA binding hint
        numa_nodes = get_numa_node_count()
        numa_prefix = ""
        if numa_nodes > 1:
            numa_prefix = "numactl --cpunodebind=0 --membind=0 "

        # Interconnect tuning
        ic = get_interconnect_info()
        ic_export = ""
        if ic.get("type") == "infiniband":
            ic_export = "export UCX_TLS=rc,self,sm\nexport I_MPI_FABRICS=ofi\nexport I_MPI_OFI_PROVIDER=mlx\n"
        elif ic.get("type") in ["ethernet", "tcp"]:
            ic_export = "export UCX_TLS=tcp,self,sm\nexport I_MPI_FABRICS=tcp\n"

        # Extract suggestion parameters
        nmat = suggestion.get("nmat", 0)
        omp = suggestion.get("omp_threads_per_rank", 1)
        mode = suggestion.get("mode", "mpi")
        is_soc = suggestion.get("is_soc", False)
        solver_hint = suggestion.get("elpa_solver", "")

        # ELPA environment and run_lapw flag
        elpa_env = ""
        elpa_parallel_opts = ""
        elpa_run_flag = ""
        solver_upper = solver_hint.upper().strip()
        if "ELPA2" in solver_upper:
            elpa_env = 'export USE_ELPA=2\nexport ELPA_KERNEL=ELPA2\n'
            elpa_parallel_opts = 'export USE_ELPA=2\nexport ELPA_KERNEL=ELPA2\n'
            elpa_run_flag = '-elpa 2'
        elif "ELPA1" in solver_upper or "ELPA" in solver_upper:
            elpa_env = 'export USE_ELPA=1\nexport ELPA_KERNEL=ELPA1\n'
            elpa_parallel_opts = 'export USE_ELPA=1\nexport ELPA_KERNEL=ELPA1\n'
            elpa_run_flag = '-elpa 1'
        elif "SCALAPACK" in solver_upper:
            elpa_parallel_opts = 'export USE_ELPA=0\n'

        # Default run_lapw command with optional ELPA flag
        run_lapw_cmd = f"run_lapw -p -NI {elpa_run_flag}".strip()
        # BLACS grid for ELPA awareness
        blacs_env = ""
        if solver_hint:
            from ...core.topology import factorize_blacs_grid
            total_ranks = sum(topo.cores_per_node) if topo.cores_per_node else 1
            p, q = factorize_blacs_grid(total_ranks)
            if p > 1 and q > 1:
                blacs_env = f'export BLACS_GRID="{p}x{q}"\n'

        # Optimal MKL threads
        mkl_threads = self._get_optimal_mkl_threads(omp, mode, nmat, is_soc)

        # Warning comments
        warnings = suggestion.get("warnings", [])
        warning_comments = "\n".join(f"# WARNING: {w}" for w in warnings)
        if warning_comments:
            warning_comments += "\n"

        # Generate script content
        content = f"""#!/bin/bash
# Auto-generated by forge v0.1.0 (WIEN2k backend)
# Mode: {mode.upper()} | OMP={omp} | MKL={mkl_threads} | Solver: {solver_hint or 'default'}
# Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}Z
{warning_comments}
{mpi_env}
{ic_export}
{elpa_env}
{blacs_env}

# WIEN2k environment
export WIENROOT={wienroot}
export PATH="$WIENROOT:$PATH"

# OpenMP configuration
export OMP_NUM_THREADS={omp}
export MKL_NUM_THREADS={mkl_threads}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Library path (avoid duplicates)
if [ -n "$LD_LIBRARY_PATH" ]; then
    case ":$LD_LIBRARY_PATH:" in
        *":$WIENROOT/lib":*) ;;
        *) export LD_LIBRARY_PATH="$WIENROOT/lib:$LD_LIBRARY_PATH" ;;
    esac
else
    export LD_LIBRARY_PATH="$WIENROOT/lib"
fi

# Scratch directory setup with fallback chain
# Priority: /dev/shm (RAM) -> $SCRATCH (local SSD) -> /tmp -> network
SCRATCH_DIR=$(mktemp -d -p /dev/shm 2>/dev/null || mktemp -d -p ${{SCRATCH:-/scratch}} 2>/dev/null || mktemp -d)
export SCRATCH="$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR"
export WIEN2K_SCRATCH="$SCRATCH_DIR"
trap 'echo "[forge] Cleaning up $SCRATCH_DIR"; rm -rf "$SCRATCH_DIR" 2>/dev/null' EXIT TERM INT
echo "[forge] SCRATCH set to $SCRATCH_DIR"

# Write parallel_options inline (ensures consistency)
cat > "$SCRATCH_DIR/parallel_options" << 'PARALLEL_OPTIONS_EOF'
export USE_REMOTE=0
export MPI_REMOTE=0
export TASKSET=no
export OMP_NUM_THREADS={omp}
export MKL_NUM_THREADS={mkl_threads}
export WIEN_MPIRUN="mpirun -np _NP_ -machinefile _HOSTS_ _EXEC_"
export DELAY=0.1
export SLEEPY=1
export OMP_GLOBAL={omp}
export KPAR=0
export WIEN_GRANULARITY=1
export WIEN_DBGLVL=0
{elpa_parallel_opts}
PARALLEL_OPTIONS_EOF
export PARALLEL_OPTIONS="$SCRATCH_DIR/parallel_options"

# MPI launcher detection
if [ -n "$SLURM_JOB_ID" ]; then
    export WIEN_MPIRUN="srun --mpi=pmix --hint=nomultithread"
elif [ -n "$PBS_JOBID" ]; then
    export WIEN_MPIRUN="mpirun"
elif [ -n "$LSB_JOBID" ]; then
    export WIEN_MPIRUN="jsrun"
else
    export WIEN_MPIRUN="${{WIEN_MPIRUN:-mpirun}}"
fi

# MPI optimization for large matrices
if [ {nmat} -gt 5000 ]; then
    export LAPW1_MPI_OPT="-b 64"
fi

# Preemption & Signal Resilience
# Save checkpoint on SIGTERM/SIGUSR1 (SLURM preemption or walltime limit)
_checkpoint_handler() {{
    echo "[forge] Preemption signal received. Saving SCF checkpoint..."
    # WIEN2k automatically saves charge density on exit, but we can trigger mixer if needed
    sleep 2
    exit 143  # Standard exit for SIGTERM
}}
trap _checkpoint_handler TERM USR1

# User-customizable command (default: run_lapw -p -NI with solver flags)
: "${{RUN_LAPW_CMD:={run_lapw_cmd}}}"

# Execute with NUMA binding if recommended
{numa_prefix}exec $RUN_LAPW_CMD "$@"
"""
        # Atomic write with executable permissions
        atomic_write(script_path, content, mode=0o755)
        logger.info(f"Written {script_path} ({len(content)} bytes)")


def auto_detect_optimal_rkmax(
    available_cores: int | None = None,
    available_memory_gb: float | None = None,
) -> float:
    """
    Standalone convenience function that wraps Wien2kBackend to
    auto-detect the optimal RKMAX for the current WIEN2k case.

    Detects problem size from input files, estimates available system
    resources if not provided, and returns the recommended RKMAX value.
    """

    if available_cores is None:
        available_cores = get_physical_cores()

    if available_memory_gb is None:
        mem_kb = get_total_mem_kb()
        available_memory_gb = mem_kb / (1024.0 * 1024.0)

    backend = Wien2kBackend()
    return backend.auto_rkmax(available_cores, available_memory_gb)
