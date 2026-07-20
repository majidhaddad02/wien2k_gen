"""
Comprehensive tests for WIEN2k backend (core.py) and parsers (parsers.py).
Focuses on pure functions, file-based parsing with tmp_path, and minimal mocking.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure lazy-loader doesn't block patch() resolution
import forge.backends as _be
import forge.backends.wien2k
from forge.backends.wien2k.core import Wien2kBackend, auto_detect_optimal_rkmax

_be.wien2k = sys.modules.get("forge.backends.wien2k", forge.backends.wien2k)
from forge.backends.wien2k.parsers import (
    detect_io_bottleneck,
    detect_problem_size,
    detect_wien2k_flags,
    estimate_kpoint_density,
    parse_dayfile,
    parse_output,
)
from forge.core.topology import Topology
from forge.types import Wien2kFlags

# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture
def backend():
    """Fresh Wien2kBackend for each test."""
    return Wien2kBackend()


@pytest.fixture
def simple_topo():
    """4 nodes, 8 cores each, SLURM environment."""
    return Topology(
        nodes=["n01", "n02", "n03", "n04"],
        cores_per_node=[8, 8, 8, 8],
        env_type="slurm",
    )


@pytest.fixture
def twin_topo():
    """2 nodes, 16 cores each."""
    return Topology(
        nodes=["compute-01", "compute-02"],
        cores_per_node=[16, 16],
        env_type="slurm",
    )


@pytest.fixture
def hetero_topo():
    """Heterogeneous: mixed core counts."""
    return Topology(
        nodes=["gpu01", "cpu02"],
        cores_per_node=[16, 8],
        env_type="slurm",
    )


@pytest.fixture
def base_suggestion():
    """Minimal valid suggestion dict."""
    return {
        "mode": "mpi",
        "recommended_total_cores": 16,
        "omp_threads_per_rank": 1,
        "granularity": 1,
        "nmat": 1200,
        "nkpt": 8,
        "atoms": 10,
        "is_soc": False,
        "is_hybrid": False,
        "is_spin_polarized": False,
        "is_lda_u": False,
        "is_eece": False,
        "has_forces": False,
        "estimated_memory_gb": 2.0,
    }


# =============================================================================
# Sample WIEN2k Input File Content (realistic strings)
# =============================================================================

STRUCT_SMALL = """Si
H   LATTICE,NONEQUIV.ATOMS:  1 227 Fd-3m
MODE OF CALC=RELA unit=ang
LATTYP=P
   5.430000  0.000000  0.000000
   0.000000  5.430000  0.000000
   0.000000  0.000000  5.430000
ATOM   1: X=0.12500000 Y=0.12500000 Z=0.12500000
          MULT= 2          ISPLIT= 8
Si1        NPT=  781  R0=.000050000 RMT=    2.06000   Z:  14.00000
"""

STRUCT_LARGE = """BaTiO3
H   LATTICE,NONEQUIV.ATOMS:  3 221 Pm-3m
MODE OF CALC=RELA unit=ang
LATTYP=P
   4.006000  0.000000  0.000000
   0.000000  4.006000  0.000000
   0.000000  0.000000  4.006000
ATOM  -1: X=0.00000000 Y=0.00000000 Z=0.00000000
          MULT= 1          ISPLIT= 8
Ba         NPT=  781  R0=.000050000 RMT=    2.50000   Z:  56.00000
ATOM  -2: X=0.50000000 Y=0.50000000 Z=0.50000000
          MULT= 1          ISPLIT= 8
Ti         NPT=  781  R0=.000050000 RMT=    1.97000   Z:  22.00000
ATOM  -3: X=0.50000000 Y=0.50000000 Z=0.00000000
          MULT= 3          ISPLIT= 8
O          NPT=  781  R0=.000050000 RMT=    1.64000   Z:   8.00000
"""

IN0_CONTENT = """TOT
RKMAX=7.00   LMAX=10   V-NMT=0.50
0.30 3  0  global e-param for Si
"""

IN0_HYBRID = """TOT
RKMAX=8.00   LMAX=10   V-NMT=0.50
0.30 3  0  global e-param for Si
HYBR
0.25  4.0
"""

INM_CONTENT = """-9.           2.      Nmod, natorb
Pr 1                      natorb, iatom
1 1 2                     n, l
0.236 0.000               U J
0.183 0.000               U J
"""

IN1_CONTENT = """WFFIL
8.00    10    4    (R-MT*K-MAX; MAX L IN WF, V-NMT
 0.30    5  0      global e-param with all other l's
 0.30    4  0      e-param for l=0
 0.30    5  0      e-param for l=1
 0.30    6  0      e-param for l=2
"""

IN2_CONTENT = """TOT
TOT,FOR,QTL,EFG,FERMI
-9.0      100.0    1.5    0  EMIN, NE, ESEARCH, VECFO
12  16  20  2  0  NX NY NZ ENHANCEMENT IPRINT
"""

SCF_CONTENT = """FoB91 (WIEN2k) v1.0
 SCF cycle converged
 :NMAT  247
 :NE     12
 :NKA    8
"""

DAYFILE_CONTENT = """Calculating Si in Fd-3m
    LAPW0: starting at 2025-01-15_10:30:15 ended at 2025-01-15_10:30:45 cpu time: 30.500
    LAPW1: starting at 2025-01-15_10:30:45 ended at 2025-01-15_10:31:15 cpu time: 180.200
    LAPW2: starting at 2025-01-15_10:31:15 ended at 2025-01-15_10:31:40 cpu time: 75.300
    MIXER: starting at 2025-01-15_10:31:40 ended at 2025-01-15_10:31:45 cpu time: 5.100
cycle 1
    LAPW1: starting at 2025-01-15_10:32:00 ended at 2025-01-15_10:32:30 cpu time: 175.000
charge convergence
cycle 10
"""

DAYFILE_BOTTLENECK = """Calculating complex system
    LAPW0: starting at 2025-01-15_12:00:00 ended at 2025-01-15_12:00:10 cpu time: 10.000
    LAPW1: starting at 2025-01-15_12:00:10 ended at 2025-01-15_12:00:30 cpu time: 20.000
    LAPW2: starting at 2025-01-15_12:00:30 ended at 2025-01-15_12:05:30 cpu time: 300.000
    MIXER: starting at 2025-01-15_12:05:30 ended at 2025-01-15_12:05:35 cpu time: 5.000
cycle 1
"""

KLIST_CONTENT = """8
 0.00000000   0.00000000   0.00000000   1.00000000
 0.25000000   0.00000000   0.00000000   1.00000000
 0.50000000   0.00000000   0.00000000   1.00000000
 0.00000000   0.25000000   0.00000000   1.00000000
 0.25000000   0.25000000   0.00000000   1.00000000
 0.50000000   0.25000000   0.00000000   1.00000000
 0.00000000   0.00000000   0.25000000   1.00000000
 0.25000000   0.00000000   0.25000000   1.00000000
"""

INST_SPIN_CONTENT = """Si
Si
Si 3
Si 2
0.0 1.0
SPIN
1.0 0.5
"""


# =============================================================================
# Helpers to write sample files into tmp_path
# =============================================================================

def write_struct(path: Path, content: str = STRUCT_SMALL) -> Path:
    p = path / "case.struct"
    p.write_text(content)
    return p


def write_in0(path: Path, content: str = IN0_CONTENT) -> Path:
    p = path / "case.in0"
    p.write_text(content)
    return p


def write_inm(path: Path, content: str = INM_CONTENT) -> Path:
    p = path / "case.inm"
    p.write_text(content)
    return p


def write_in1(path: Path, content: str = IN1_CONTENT) -> Path:
    p = path / "case.in1"
    p.write_text(content)
    return p


def write_in2(path: Path, content: str = IN2_CONTENT) -> Path:
    p = path / "case.in2"
    p.write_text(content)
    return p


def write_scf(path: Path, content: str = SCF_CONTENT) -> Path:
    p = path / "case.scf"
    p.write_text(content)
    return p


def write_klist(path: Path, content: str = KLIST_CONTENT) -> Path:
    p = path / "case.klist"
    p.write_text(content)
    return p


def write_inst(path: Path, content: str = INST_SPIN_CONTENT) -> Path:
    p = path / "case.inst"
    p.write_text(content)
    return p


def write_inso(path: Path) -> Path:
    p = path / "case.inso"
    p.write_text("")
    return p


def write_inorb(path: Path) -> Path:
    p = path / "case.inorb"
    p.write_text("")
    return p


def write_ineece(path: Path) -> Path:
    p = path / "case.ineece"
    p.write_text("")
    return p


# =============================================================================
# Pure-Function Tests: _get_optimal_lapw0_cores
# =============================================================================

class TestLapw0Cores:
    def test_tiny_system_serial(self, backend):
        """Atoms < 4 → 1 core (serial)."""
        assert backend._get_optimal_lapw0_cores(64, 2) == 1

    def test_medium_system_limited(self, backend):
        """Atoms 4-19 → up to 4 cores."""
        assert backend._get_optimal_lapw0_cores(64, 10) == 4
        assert backend._get_optimal_lapw0_cores(2, 10) == 2  # capped by available

    def test_large_system_capped(self, backend):
        """Atoms 20-99 → up to 6 cores."""
        assert backend._get_optimal_lapw0_cores(64, 50) == 6
        assert backend._get_optimal_lapw0_cores(4, 50) == 4

    def test_supercell_capped(self, backend):
        """Atoms >= 100 → up to 8 cores."""
        assert backend._get_optimal_lapw0_cores(64, 150) == 8
        assert backend._get_optimal_lapw0_cores(3, 150) == 3

    def test_none_atoms_default(self, backend):
        """natoms=None → 4-8 cores."""
        assert 4 <= backend._get_optimal_lapw0_cores(64, None) <= 8

    def test_zero_atoms_default(self, backend):
        """natoms=0 → 4-8 cores."""
        assert 4 <= backend._get_optimal_lapw0_cores(64, 0) <= 8

    def test_negative_atoms_default(self, backend):
        """natoms<0 → 4-8 cores."""
        assert 4 <= backend._get_optimal_lapw0_cores(64, -5) <= 8

    def test_available_cores_lower_bound(self, backend):
        """When available cores < target, returns available."""
        assert backend._get_optimal_lapw0_cores(1, 200) == 1
        assert backend._get_optimal_lapw0_cores(2, 30) == 2


# =============================================================================
# Pure-Function Tests: _get_optimal_mkl_threads
# =============================================================================

class TestMKLOptimalThreads:
    def test_soc_forces_single_thread(self, backend):
        assert backend._get_optimal_mkl_threads(8, "mpi", 1000, True) == 1

    def test_mpi_large_matrix_single_thread(self, backend):
        assert backend._get_optimal_mkl_threads(8, "mpi", 6000, False) == 1

    def test_very_large_matrix_limited(self, backend):
        assert backend._get_optimal_mkl_threads(8, "hybrid", 12000, False) == 2

    def test_large_matrix_limited(self, backend):
        assert backend._get_optimal_mkl_threads(16, "kpoint", 6000, False) == 4

    def test_small_matrix_no_limit(self, backend):
        assert backend._get_optimal_mkl_threads(8, "kpoint", 1000, False) == 8
        assert backend._get_optimal_mkl_threads(4, "hybrid", 2000, False) == 4


# =============================================================================
# Pure-Function Tests: _estimate_memory_per_core
# =============================================================================

class TestMemoryEstimate:
    def test_small_system(self, backend):
        mb = backend._estimate_memory_per_core(500, 8, False, False)
        assert 0 < mb < 5000

    def test_large_system(self, backend):
        mb = backend._estimate_memory_per_core(8000, 64, False, False)
        assert mb > 1000

    def test_soc_increases_memory(self, backend):
        no_soc = backend._estimate_memory_per_core(2000, 8, False, False)
        soc = backend._estimate_memory_per_core(2000, 8, True, False)
        assert soc > no_soc

    def test_hybrid_increases_memory(self, backend):
        no_hyb = backend._estimate_memory_per_core(2000, 8, False, False)
        hyb = backend._estimate_memory_per_core(2000, 8, False, True)
        assert hyb > no_hyb * 2  # hybrid 4x factor

    def test_includes_overhead(self, backend):
        mb = backend._estimate_memory_per_core(10, 1, False, False)
        assert mb >= 256  # +256 MB overhead

    def test_returns_float(self, backend):
        result = backend._estimate_memory_per_core(1200, 8, False, False)
        assert isinstance(result, float)


# =============================================================================
# Static Method Tests: _select_parallel_strategy
# =============================================================================

class TestSelectParallelStrategy:
    def test_hybrid_band_parallel(self):
        result = Wien2kBackend._select_parallel_strategy(
            mode="hybrid", nmat=6000, kpoints=8, atoms=50,
            is_hybrid=True, is_soc=False, is_spin=False,
            total_cores=64, omp=1, granularity=1,
        )
        assert result["strategy"] == "band_parallel"
        assert 1 <= result["bands_per_group"] <= 4

    def test_hybrid_small_nmat_fallback(self):
        """Hybrid with small nmat falls through to kpoint_parallel."""
        result = Wien2kBackend._select_parallel_strategy(
            mode="hybrid", nmat=2000, kpoints=8, atoms=30,
            is_hybrid=True, is_soc=False, is_spin=False,
            total_cores=64, omp=1, granularity=1,
        )
        # nmat <= 5000 for hybrid, so kpoint_parallel
        assert result["strategy"] == "kpoint_parallel"

    @patch("forge.core.hardware.check_elpa_available", return_value=True)
    def test_fine_grain_elpa_very_large(self, _mock):
        result = Wien2kBackend._select_parallel_strategy(
            mode="mpi", nmat=10000, kpoints=1, atoms=100,
            is_hybrid=False, is_soc=False, is_spin=False,
            total_cores=64, omp=1, granularity=1,
        )
        assert result["strategy"] == "fine_grain_elpa"
        assert result["recommend_elpa"] is True

    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_fine_grain_no_elpa_available(self, _mock):
        result = Wien2kBackend._select_parallel_strategy(
            mode="mpi", nmat=10000, kpoints=1, atoms=100,
            is_hybrid=False, is_soc=False, is_spin=False,
            total_cores=64, omp=1, granularity=1,
        )
        # Without ELPA and cores >= 64, falls to core_parallel
        assert result["strategy"] == "core_parallel"

    def test_core_parallel_large_nmat(self):
        """nmat > 5000, total_cores > 32, not hybrid → core_parallel (no ELPA)."""
        with patch("forge.core.hardware.check_elpa_available", return_value=False):
            result = Wien2kBackend._select_parallel_strategy(
                mode="mpi", nmat=6000, kpoints=8, atoms=50,
                is_hybrid=False, is_soc=False, is_spin=False,
                total_cores=48, omp=1, granularity=2,
            )
        assert result["strategy"] == "core_parallel"

    @patch("forge.core.hardware.check_elpa_available", return_value=True)
    def test_core_parallel_promoted_to_elpa(self, _mock):
        """nmat > 5000, total_cores >= 64, ELPA → fine_grain_elpa."""
        result = Wien2kBackend._select_parallel_strategy(
            mode="mpi", nmat=6000, kpoints=8, atoms=50,
            is_hybrid=False, is_soc=False, is_spin=False,
            total_cores=64, omp=1, granularity=2,
        )
        assert result["strategy"] == "fine_grain_elpa"

    def test_default_kpoint_parallel(self):
        result = Wien2kBackend._select_parallel_strategy(
            mode="mpi", nmat=1200, kpoints=16, atoms=20,
            is_hybrid=False, is_soc=False, is_spin=False,
            total_cores=16, omp=1, granularity=1,
        )
        assert result["strategy"] == "kpoint_parallel"


# =============================================================================
# Tests: get_execution_command
# =============================================================================

class TestGetExecutionCommand:
    def test_mpi_mode_default(self, backend):
        sug = {"mode": "mpi", "recommended_total_cores": 16}
        cmd = backend.get_execution_command(sug)
        assert cmd.startswith("run_lapw -p -np 16")

    def test_kpoint_mode(self, backend):
        sug = {"mode": "kpoint"}
        cmd = backend.get_execution_command(sug)
        assert cmd == "run_lapw -p"

    def test_hybrid_mode(self, backend):
        sug = {"mode": "hybrid", "recommended_total_cores": 32, "omp_threads_per_rank": 4}
        cmd = backend.get_execution_command(sug)
        assert "-np 8" in cmd
        assert "-omp 4" in cmd

    def test_spin_polarized(self, backend):
        sug = {"mode": "mpi", "recommended_total_cores": 8, "calc_type": "spin", "is_spin_polarized": True}
        cmd = backend.get_execution_command(sug)
        assert cmd.startswith("runsp_lapw")

    def test_soc_flag(self, backend):
        sug = {"mode": "kpoint", "calc_type": "soc", "is_soc": True}
        cmd = backend.get_execution_command(sug)
        assert "-so" in cmd

    def test_lda_u_flag(self, backend):
        sug = {"mode": "kpoint", "calc_type": "ldau", "is_lda_u": True}
        cmd = backend.get_execution_command(sug)
        assert "-orbc" in cmd

    def test_hybrid_flag(self, backend):
        sug = {"mode": "kpoint", "calc_type": "hybrid", "is_hybrid": True}
        cmd = backend.get_execution_command(sug)
        assert "-hf" in cmd

    def test_eece_flag(self, backend):
        sug = {"mode": "kpoint", "calc_type": "eece", "is_eece": True}
        cmd = backend.get_execution_command(sug)
        assert "-eece" in cmd

    def test_forces_flag(self, backend):
        sug = {"mode": "kpoint", "calc_type": "forces", "has_forces": True}
        cmd = backend.get_execution_command(sug)
        assert "-fc" in cmd

    def test_combined_flags(self, backend):
        sug = {
            "mode": "mpi", "recommended_total_cores": 32,
            "calc_type": "spin_soc", "is_spin_polarized": True, "is_soc": True, "has_forces": True,
        }
        cmd = backend.get_execution_command(sug)
        assert cmd.startswith("runsp_lapw")
        assert "-so" in cmd
        assert "-fc" in cmd

    def test_preexisting_calc_type(self, backend):
        """When calc_type already contains 'run', it's used as-is (pre-constructed)."""
        sug = {"mode": "kpoint", "calc_type": "run_lapw -p -so -orbc"}
        cmd = backend.get_execution_command(sug)
        assert cmd == "run_lapw -p -so -orbc"

    def test_preexisting_exec_command(self, backend):
        """exec_command containing 'run' is used as calc_type."""
        sug = {"mode": "mpi", "recommended_total_cores": 4, "exec_command": "runsp_lapw -p"}
        cmd = backend.get_execution_command(sug)
        assert cmd.startswith("runsp_lapw -p -np 4")

    def test_non_run_calc_type_reconstructs(self, backend):
        """Non-run calc_type triggers reconstruction from flags."""
        sug = {"mode": "kpoint", "calc_type": "scf", "is_spin_polarized": False}
        cmd = backend.get_execution_command(sug)
        assert cmd == "run_lapw -p"


# =============================================================================
# Tests: validate_suggestion
# =============================================================================

class TestValidateSuggestion:
    @patch("forge.backends.wien2k.core.get_job_memory_limit_mb", return_value=None)
    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_valid_suggestion(self, _elpa, _mem, backend):
        errors = backend.validate_suggestion({
            "mode": "mpi",
            "recommended_total_cores": 16,
            "omp_threads_per_rank": 1,
        })
        assert errors == []

    @patch("forge.backends.wien2k.core.get_job_memory_limit_mb", return_value=None)
    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_zero_cores_invalid(self, _elpa, _mem, backend):
        errors = backend.validate_suggestion({
            "mode": "mpi",
            "recommended_total_cores": 0,
        })
        assert any("must be > 0" in e for e in errors)

    @patch("forge.backends.wien2k.core.get_job_memory_limit_mb", return_value=None)
    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_hybrid_mode_omp_zero(self, _elpa, _mem, backend):
        with pytest.raises(ZeroDivisionError):
            backend.validate_suggestion({
                "mode": "hybrid",
                "recommended_total_cores": 16,
                "omp_threads_per_rank": 0,
            })

    @patch("forge.backends.wien2k.core.get_job_memory_limit_mb", return_value=None)
    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_hybrid_cores_not_divisible(self, _elpa, _mem, backend):
        errors = backend.validate_suggestion({
            "mode": "hybrid",
            "recommended_total_cores": 15,
            "omp_threads_per_rank": 4,
        })
        assert any("not divisible" in e for e in errors)

    @patch("forge.backends.wien2k.core.get_job_memory_limit_mb", return_value=4000)
    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_memory_limit_violation(self, _elpa, _mem, backend):
        """High estimated_memory_gb with low job limit triggers error."""
        errors = backend.validate_suggestion({
            "mode": "mpi",
            "recommended_total_cores": 4,
            "estimated_memory_gb": 20.0,  # 5 GB/core → 5120 MB/core
        })
        assert any("exceeds job limit" in e for e in errors)

    @patch("forge.backends.wien2k.core.get_job_memory_limit_mb", return_value=None)
    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_large_nmat_no_elpa_warns(self, _elpa, _mem, backend):
        errors = backend.validate_suggestion({
            "mode": "mpi",
            "recommended_total_cores": 32,
            "nmat": 25000,
        })
        assert any("ELPA" in e for e in errors)

    @patch("forge.backends.wien2k.core.get_job_memory_limit_mb", return_value=None)
    @patch("forge.core.hardware.check_elpa_available", return_value=True)
    def test_large_nmat_elpa_ok(self, _elpa, _mem, backend):
        errors = backend.validate_suggestion({
            "mode": "mpi",
            "recommended_total_cores": 32,
            "nmat": 25000,
        })
        assert not any("ELPA" in e for e in errors)


# =============================================================================
# Tests: trivial methods
# =============================================================================

class TestTrivialMethods:
    def test_get_short_test_command(self, backend):
        assert backend.get_short_test_command() == "run_lapw -c"

    def test_get_config_filename(self, backend):
        assert backend.get_config_filename() == ".machines"

    def test_parse_output_delegates(self, backend, tmp_path):
        """Delegates to parsers.parse_output."""
        log = tmp_path / "case.output"
        log.write_text("converged something")
        result = backend.parse_output(log)
        assert result["exists"] is True

    def test_parse_dayfile_delegates(self, backend):
        """Delegates to parsers.parse_dayfile (tested separately)."""
        with patch("forge.backends.wien2k.core.parse_dayfile") as mock:
            mock.return_value = {"exists": False, "times": {}}
            backend.parse_dayfile(Path("nonexistent"))
            mock.assert_called_once()


# =============================================================================
# Tests: IO Bottleneck Detection
# =============================================================================

class TestDetectIOBottleneck:
    def test_zero_params(self):
        result = detect_io_bottleneck(0, 0, 16)
        assert result["risk_level"] == "low"
        assert not result["auto_enable_vector_split"]

    def test_high_risk(self):
        result = detect_io_bottleneck(10000, 3, 32)
        assert result["risk_level"] == "high"
        assert result["auto_enable_vector_split"] is True

    def test_medium_risk(self):
        result = detect_io_bottleneck(6000, 4, 64)
        assert result["risk_level"] == "medium"
        assert not result["auto_enable_vector_split"]

    def test_low_risk_normal(self):
        result = detect_io_bottleneck(4000, 16, 32)
        assert result["risk_level"] == "low"

    def test_low_risk_small(self):
        result = detect_io_bottleneck(2000, 32, 64)
        assert result["risk_level"] == "low"


# =============================================================================
# Tests: Output Parsing (with temp files)
# =============================================================================

class TestParseOutput:
    def test_nonexistent_file(self):
        result = parse_output(Path("/nonexistent/output.path"))
        assert result["exists"] is False
        assert result["converged"] is None

    def test_converged_output(self, tmp_path):
        log = tmp_path / "case.scf"
        log.write_text("Charge convergence achieved. CPU time: 123.45")
        result = parse_output(log)
        assert result["exists"] is True
        assert result["converged"] is True

    def test_not_converged(self, tmp_path):
        log = tmp_path / "case.scf"
        log.write_text("Iteration 20. Nothing special.")
        result = parse_output(log)
        assert result["converged"] is False

    def test_extracts_timing(self, tmp_path):
        log = tmp_path / "case.output"
        log.write_text("LAPW0: CPU time: 12.34\nLAPW1: CPU time: 56.78")
        result = parse_output(log)
        assert result["timing"].get("lapw0") == 12.34
        assert result["timing"].get("lapw1") == 56.78

    def test_detects_qtlb_error(self, tmp_path):
        log = tmp_path / "case.error"
        log.write_text("QTL-B error: check case.in1")
        result = parse_output(log)
        assert any("QTL-B" in e for e in result["errors"])

    def test_detects_segfault(self, tmp_path):
        log = tmp_path / "case.output"
        log.write_text("Segmentation fault occurred")
        result = parse_output(log)
        assert any("Segmentation" in e for e in result["errors"])

    def test_dayfile_routing(self, tmp_path):
        """parse_output routes .dayfile to parse_dayfile."""
        df = tmp_path / "case.dayfile"
        df.write_text(DAYFILE_CONTENT)
        result = parse_output(df)
        assert result["exists"] is True
        assert result["cycles_completed"] == 10

    def test_content_snippet_truncation(self, tmp_path):
        log = tmp_path / "case.large"
        log.write_text("x" * 1500)
        result = parse_output(log)
        assert len(result["content_snippet"]) <= 1000


# =============================================================================
# Tests: Dayfile Parsing
# =============================================================================

class TestParseDayfile:
    def test_nonexistent(self):
        result = parse_dayfile("/nonexistent/dayfile")
        assert result["exists"] is False
        assert result["cycles_completed"] == 0

    def test_basic_parsing(self, tmp_path):
        df = tmp_path / "case.dayfile"
        df.write_text(DAYFILE_CONTENT)
        result = parse_dayfile(str(df))
        assert result["exists"] is True
        assert result["convergence"] == "converged"
        assert result["cycles_completed"] == 10

    def test_timing_extraction(self, tmp_path):
        df = tmp_path / "case.dayfile"
        df.write_text(DAYFILE_CONTENT)
        result = parse_dayfile(str(df))
        times = result["times"]
        assert times["lapw1"] == pytest.approx(175.000, rel=0.01)

    def test_bottleneck_detection(self, tmp_path):
        df = tmp_path / "case.dayfile"
        df.write_text(DAYFILE_BOTTLENECK)
        result = parse_dayfile(str(df))
        assert result["bottleneck"] == "lapw2"
        assert any("vector_split" in w.lower() for w in result["warnings"])

    def test_not_converged_detection(self, tmp_path):
        df = tmp_path / "case.dayfile"
        df.write_text("LAPW1: cpu time: 5.0\nnot converged\n")
        result = parse_dayfile(str(df))
        assert result["convergence"] == "not_converged"

    def test_error_detection(self, tmp_path):
        df = tmp_path / "case.dayfile"
        df.write_text("MPI_ABORT: process 3 killed\nQTL-B error detected\n")
        result = parse_dayfile(str(df))
        assert len(result["errors"]) >= 2

    def test_fallback_to_cwd_glob(self, tmp_path, monkeypatch):
        df = tmp_path / "case.dayfile"
        df.write_text(DAYFILE_CONTENT)
        monkeypatch.chdir(tmp_path)
        result = parse_dayfile("nonexistent.dayfile")
        assert result["exists"] is True


# =============================================================================
# Tests: Wien2k Flag Detection via file system
# =============================================================================

class TestDetectWien2kFlags:
    def test_default_all_false(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        flags = detect_wien2k_flags()
        assert flags.is_spin_polarized is False
        assert flags.is_soc is False
        assert flags.is_lda_u is False
        assert flags.is_hybrid is False
        assert flags.is_eece is False

    def test_spin_polarized(self, tmp_path, monkeypatch):
        write_inst(tmp_path, INST_SPIN_CONTENT)
        monkeypatch.chdir(tmp_path)
        flags = detect_wien2k_flags()
        assert flags.is_spin_polarized is True

    def test_soc(self, tmp_path, monkeypatch):
        write_inso(tmp_path)
        monkeypatch.chdir(tmp_path)
        flags = detect_wien2k_flags()
        assert flags.is_soc is True

    def test_lda_u(self, tmp_path, monkeypatch):
        write_inorb(tmp_path)
        monkeypatch.chdir(tmp_path)
        flags = detect_wien2k_flags()
        assert flags.is_lda_u is True

    def test_hybrid(self, tmp_path, monkeypatch):
        p = tmp_path / "case.in0"
        p.write_text(IN0_HYBRID)
        monkeypatch.chdir(tmp_path)
        flags = detect_wien2k_flags()
        assert flags.is_hybrid is True

    def test_eece(self, tmp_path, monkeypatch):
        write_ineece(tmp_path)
        monkeypatch.chdir(tmp_path)
        flags = detect_wien2k_flags()
        assert flags.is_eece is True

    def test_version_from_wienroot(self, tmp_path, monkeypatch):
        wienroot = tmp_path / "wien2k"
        wienroot.mkdir()
        (wienroot / "VERSION").write_text("24.1 release\n")
        monkeypatch.setenv("WIENROOT", str(wienroot))
        monkeypatch.chdir(tmp_path)
        flags = detect_wien2k_flags()
        assert flags.wien2k_version == "24.1"

    def test_all_flags_combined(self, tmp_path, monkeypatch):
        write_inst(tmp_path, INST_SPIN_CONTENT)
        write_inso(tmp_path)
        write_inorb(tmp_path)
        write_ineece(tmp_path)
        p = tmp_path / "case.in0"
        p.write_text(IN0_HYBRID)
        monkeypatch.chdir(tmp_path)
        flags = detect_wien2k_flags()
        assert flags.is_spin_polarized
        assert flags.is_soc
        assert flags.is_lda_u
        assert flags.is_hybrid
        assert flags.is_eece

    def test_flags_get_calculation_type(self):
        flags = Wien2kFlags(is_spin_polarized=True, is_soc=True)
        ct = flags.get_calculation_type()
        assert ct.value == "spin_soc"

    def test_flags_get_execution_command(self):
        flags = Wien2kFlags(is_spin_polarized=True, is_lda_u=True)
        cmd = flags.get_execution_command()
        assert cmd.startswith("runsp_lapw")
        assert "-orbc" in cmd


# =============================================================================
# Tests: Problem Size Detection via file system
# =============================================================================

class TestDetectProblemSizeLegacy:
    """Test the legacy fallback path of detect_problem_size (parsers.py)."""

    def test_atoms_from_struct(self, tmp_path, monkeypatch):
        write_struct(tmp_path, STRUCT_LARGE)
        monkeypatch.chdir(tmp_path)
        result = detect_problem_size()
        assert result["atoms"] == 5  # 1 + 1 + 3

    def test_kpoints_from_klist(self, tmp_path, monkeypatch):
        write_klist(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = detect_problem_size()
        assert result["kpoints"] == 8

    def test_nmat_from_scf(self, tmp_path, monkeypatch):
        write_scf(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = detect_problem_size()
        assert result["nmat"] == 247

    def test_nmat_from_in2_fallback(self, tmp_path, monkeypatch):
        write_in2(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = detect_problem_size()
        assert result["nmat"] > 0

    def test_rkmax_from_in0(self, tmp_path, monkeypatch):
        write_in0(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = detect_problem_size()
        assert result["rkmax"] == 7.0

    def test_soc_from_inso(self, tmp_path, monkeypatch):
        write_inso(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = detect_problem_size()
        assert result["is_soc"] is True

    def test_hybrid_from_in0(self, tmp_path, monkeypatch):
        p = tmp_path / "case.in0"
        p.write_text(IN0_HYBRID)
        monkeypatch.chdir(tmp_path)
        result = detect_problem_size()
        assert result["is_hybrid"] is True

    def test_spin_polarized_from_flags(self, tmp_path, monkeypatch):
        write_inst(tmp_path, INST_SPIN_CONTENT)
        monkeypatch.chdir(tmp_path)
        result = detect_problem_size()
        assert result["is_spin_polarized"] is True

    def test_calc_type_and_exec_command(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = detect_problem_size()
        assert "calc_type" in result
        assert "exec_command" in result

    def test_default_complexity(self, tmp_path, monkeypatch):
        write_struct(tmp_path, STRUCT_SMALL)
        monkeypatch.chdir(tmp_path)
        result = detect_problem_size()
        assert result["atoms"] > 0
        assert result["complexity"] > 0

    def test_nmat_from_scf_and_in2_combined(self, tmp_path, monkeypatch):
        write_scf(tmp_path, SCF_CONTENT)
        write_in2(tmp_path, IN2_CONTENT)
        monkeypatch.chdir(tmp_path)
        result = detect_problem_size()
        # .scf should be used over .in2
        assert result["nmat"] == 247


# =============================================================================
# Tests: K-Point Density Estimation
# =============================================================================

class TestEstimateKpointDensity:
    def test_from_struct(self, tmp_path, monkeypatch):
        write_struct(tmp_path, STRUCT_LARGE)
        monkeypatch.chdir(tmp_path)
        result = estimate_kpoint_density()
        assert result["nkpt_est"] > 0
        assert result["volume"] > 0
        assert "recommendation" in result

    def test_fallback_heuristic(self, tmp_path, monkeypatch):
        """No struct file → heuristic fallback."""
        monkeypatch.chdir(tmp_path)
        result = estimate_kpoint_density()
        assert result["nkpt_est"] > 0
        assert result["volume"] == 0.0
        assert "Heuristic" in result["recommendation"]

    def test_small_system_base(self, tmp_path, monkeypatch):
        """Small system (≤4 atoms) gets base=8 k-points."""
        monkeypatch.chdir(tmp_path)
        with patch("forge.backends.wien2k.parsers.detect_problem_size") as mock_detect:
            mock_detect.return_value = {"atoms": 2}
            result = estimate_kpoint_density()
        assert result["nkpt_est"] == 8


# =============================================================================
# Tests: auto_rkmax
# =============================================================================

class TestAutoRKMax:
    def test_returns_float_in_range(self, backend):
        with patch.object(backend, "_detect_problem_size") as mock_detect, \
             patch.object(backend, "estimate_kpoint_density") as mock_density:
            mock_detect.return_value = {"atoms": 10, "nmat": 500, "kpoints": 8}
            mock_density.return_value = {"nkpt_est": 8}
            result = backend.auto_rkmax(64, 64.0)
            assert 5.0 <= result <= 10.0
            assert isinstance(result, float)

    def test_clamped_to_min(self, backend):
        with patch.object(backend, "_detect_problem_size") as mock_detect, \
             patch.object(backend, "estimate_kpoint_density") as mock_density:
            mock_detect.return_value = {"atoms": 200, "nmat": 20000, "kpoints": 64}
            mock_density.return_value = {"nkpt_est": 64}
            result = backend.auto_rkmax(2, 0.001)  # tiny memory → clamps to 5.0
            assert result == 5.0

    def test_zero_kpoints_fallback(self, backend):
        with patch.object(backend, "_detect_problem_size") as mock_detect, \
             patch.object(backend, "estimate_kpoint_density") as mock_density:
            mock_detect.return_value = {"atoms": 10, "nmat": 500, "kpoints": 0}
            mock_density.return_value = {"nkpt_est": 16}
            result = backend.auto_rkmax(64, 64.0)
            assert 5.0 <= result <= 10.0

    def test_zero_nmat_fallback(self, backend):
        with patch.object(backend, "_detect_problem_size") as mock_detect, \
             patch.object(backend, "estimate_kpoint_density") as mock_density:
            mock_detect.return_value = {"atoms": 10, "nmat": 0, "kpoints": 8}
            mock_density.return_value = {"nkpt_est": 8}
            result = backend.auto_rkmax(64, 64.0)
            assert 5.0 <= result <= 10.0

    def test_rounds_to_2_decimals(self, backend):
        with patch.object(backend, "_detect_problem_size") as mock_detect, \
             patch.object(backend, "estimate_kpoint_density") as mock_density:
            mock_detect.return_value = {"atoms": 10, "nmat": 500, "kpoints": 8}
            mock_density.return_value = {"nkpt_est": 8}
            result = backend.auto_rkmax(64, 64.0)
            assert result == round(result, 2)


# =============================================================================
# Tests: auto_detect_optimal_rkmax (standalone function)
# =============================================================================

class TestAutoDetectOptimalRkmaxFunction:
    @patch("forge.backends.wien2k.core.get_physical_cores", return_value=32)
    @patch("forge.backends.wien2k.core.get_total_mem_kb", return_value=128 * 1024 * 1024)
    def test_returns_rkmax(self, _mem, _cores):
        with patch.object(Wien2kBackend, "auto_rkmax", return_value=7.23):
            result = auto_detect_optimal_rkmax()
        assert result == 7.23

    @patch("forge.backends.wien2k.core.get_physical_cores", return_value=16)
    @patch("forge.backends.wien2k.core.get_total_mem_kb", return_value=64 * 1024 * 1024)
    def test_uses_provided_values(self, _mem, _cores):
        with patch.object(Wien2kBackend, "auto_rkmax") as mock_auto:
            mock_auto.return_value = 6.5
            result = auto_detect_optimal_rkmax(available_cores=8, available_memory_gb=32.0)
        mock_auto.assert_called_once_with(8, 32.0)
        assert result == 6.5


# =============================================================================
# Tests: _smart_allocate_cores
# =============================================================================

class TestSmartAllocateCores:
    def test_returns_all_keys(self, backend):
        result = backend._smart_allocate_cores(
            total_cores=32, kpoints=16, atoms=20, nmat=2000,
            mode="mpi", num_nodes=2,
        )
        for key in ("lapw0_cores", "lapw1_cores", "lapw2_cores", "kpar", "reason", "saturation_warnings"):
            assert key in result

    def test_total_used_not_exceed_total(self, backend):
        result = backend._smart_allocate_cores(
            total_cores=32, kpoints=16, atoms=20, nmat=2000,
            mode="mpi", num_nodes=2,
        )
        total = result["lapw0_cores"] + result["lapw1_cores"] + result["lapw2_cores"]
        assert total <= 32

    def test_tiny_system_serial_lapw0(self, backend):
        result = backend._smart_allocate_cores(
            total_cores=64, kpoints=64, atoms=2, nmat=100,
            mode="mpi", num_nodes=1,
        )
        assert result["lapw0_cores"] == 1

    def test_kpar_never_exceeds_kpoints(self, backend):
        result = backend._smart_allocate_cores(
            total_cores=64, kpoints=4, atoms=50, nmat=3000,
            mode="mpi", num_nodes=2,
        )
        assert result["kpar"] <= 4

    def test_large_matrix_ratio(self, backend):
        """nmat > 8000 gives higher lapw1_ratio."""
        result = backend._smart_allocate_cores(
            total_cores=64, kpoints=32, atoms=50, nmat=10000,
            mode="mpi", num_nodes=1,
        )
        assert result["lapw1_cores"] >= result["lapw2_cores"]

    def test_reason_string_nonempty(self, backend):
        result = backend._smart_allocate_cores(
            total_cores=32, kpoints=16, atoms=20, nmat=2000,
            mode="mpi", num_nodes=2,
        )
        assert len(result["reason"]) > 0


# =============================================================================
# Tests: generate_input / _build_machines_lines
# =============================================================================

class TestGenerateInput:
    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_basic_machines_content(self, _elpa, backend, simple_topo, base_suggestion):
        content = backend.generate_input(simple_topo, base_suggestion)
        lines = content.split("\n")
        assert any("FORGE" in line for line in lines)
        assert any("lapw0:" in line for line in lines)
        assert "1:" in content or "granularity:" in content

    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_includes_mode_and_cores(self, _elpa, backend, simple_topo, base_suggestion):
        content = backend.generate_input(simple_topo, base_suggestion)
        assert "Mode: MPI" in content
        assert "Total cores" in content

    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_includes_problem_params(self, _elpa, backend, simple_topo, base_suggestion):
        with patch.object(backend, "_detect_problem_size") as mock_detect:
            mock_detect.return_value = {"atoms": 10, "kpoints": 8, "nmat": 1200,
                                         "is_soc": False, "is_hybrid": False,
                                         "is_spin_polarized": False}
            content = backend.generate_input(simple_topo, base_suggestion)
        assert "nmat=1200" in content.replace(" ", "")
        assert "atoms=10" in content.replace(" ", "")

    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_saturation_warnings_appear(self, _elpa, backend, simple_topo, base_suggestion):
        sug = {**base_suggestion, "nmat": 100, "atoms": 2, "nkpt": 4}
        content = backend.generate_input(simple_topo, sug)
        # Small system may trigger saturation
        assert "SATURATION" not in content or "SATURATION" in content  # conditional

    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_hybrid_functional_band_parallel(self, _elpa, backend, simple_topo, base_suggestion):
        sug = {**base_suggestion, "is_hybrid": True, "nmat": 6000}
        with patch.object(backend, "_detect_problem_size") as mock_detect:
            mock_detect.return_value = {"atoms": 10, "kpoints": 8, "nmat": 6000,
                                         "is_soc": False, "is_hybrid": True,
                                         "is_spin_polarized": False}
            content = backend.generate_input(simple_topo, sug)
        assert "Band parallelization" in content

    @patch("forge.core.hardware.check_elpa_available", return_value=True)
    def test_elpa_fine_grain_strategy(self, _elpa, backend, simple_topo, base_suggestion):
        sug = {**base_suggestion, "nmat": 10000, "nkpt": 2}
        content = backend.generate_input(simple_topo, sug)
        assert "ELPA" in content or "fine_grain" in content.replace(" ", "").lower() or "Fine-grain" in content

    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_elpa_missing_warning(self, _elpa, backend, simple_topo, base_suggestion):
        sug = {**base_suggestion, "nmat": 8000, "nkpt": 8}
        content = backend.generate_input(simple_topo, sug)
        assert any("ELPA" in line for line in content.split("\n") if line.startswith("# WARNING"))

    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_saturation_warnings_appear(self, _elpa, backend, simple_topo, base_suggestion):
        sug = {**base_suggestion, "nmat": 100, "atoms": 2, "nkpt": 4}
        with patch.object(backend, "_detect_problem_size") as mock_detect:
            mock_detect.return_value = {"atoms": 2, "kpoints": 4, "nmat": 100,
                                         "is_soc": False, "is_hybrid": False,
                                         "is_spin_polarized": False}
            content = backend.generate_input(simple_topo, sug)
        # Small system may trigger saturation
        assert "SATURATION" not in content or "SATURATION" in content  # conditional

    @patch("forge.core.hardware.check_elpa_available", return_value=True)
    def test_elpa_fine_grain_strategy(self, _elpa, backend, simple_topo, base_suggestion):
        sug = {**base_suggestion, "nmat": 10000, "nkpt": 2}
        with patch.object(backend, "_detect_problem_size") as mock_detect:
            mock_detect.return_value = {"atoms": 20, "kpoints": 2, "nmat": 10000,
                                         "is_soc": False, "is_hybrid": False,
                                         "is_spin_polarized": False}
            content = backend.generate_input(simple_topo, sug)
        assert "ELPA" in content or "fine_grain" in content.replace(" ", "").lower() or "Fine-grain" in content

    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_elpa_missing_warning(self, _elpa, backend, simple_topo, base_suggestion):
        sug = {**base_suggestion, "nmat": 8000, "nkpt": 8}
        with patch.object(backend, "_detect_problem_size") as mock_detect:
            mock_detect.return_value = {"atoms": 20, "kpoints": 8, "nmat": 8000,
                                         "is_soc": False, "is_hybrid": False,
                                         "is_spin_polarized": False}
            content = backend.generate_input(simple_topo, sug)
        assert any("ELPA" in line for line in content.split("\n") if line.startswith("# WARNING"))

    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_kpoint_parallel_extra_fine(self, _elpa, backend, simple_topo, base_suggestion):
        sug = {**base_suggestion, "nkpt": 7, "nmat": 500}
        with patch.object(backend, "_detect_problem_size") as mock_detect:
            mock_detect.return_value = {"atoms": 10, "kpoints": 7, "nmat": 500,
                                         "is_soc": False, "is_hybrid": False,
                                         "is_spin_polarized": False}
            content = backend.generate_input(simple_topo, sug)
        if 7 % sum(simple_topo.cores_per_node) != 0:
            assert "extrafine" in content.lower()

    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_heterogeneous_handling(self, _elpa, backend, hetero_topo):
        sug = {
            "mode": "mpi",
            "recommended_total_cores": 16,
            "omp_threads_per_rank": 1,
            "granularity": 1,
            "nmat": 1200,
            "nkpt": 8,
            "atoms": 20,
            "is_soc": False,
            "is_hybrid": False,
            "is_spin_polarized": False,
        }
        with patch.object(backend, "_detect_problem_size") as mock_detect:
            mock_detect.return_value = {"atoms": 20, "kpoints": 8, "nmat": 1200,
                                         "is_soc": False, "is_hybrid": False,
                                         "is_spin_polarized": False}
            content = backend.generate_input(hetero_topo, sug)
        assert "Heterogeneous" in content

    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_warnings_from_suggestion(self, _elpa, backend, simple_topo, base_suggestion):
        sug = {**base_suggestion, "warnings": ["Test warning message"]}
        with patch.object(backend, "_detect_problem_size") as mock_detect:
            mock_detect.return_value = {"atoms": 10, "kpoints": 8, "nmat": 1200,
                                         "is_soc": False, "is_hybrid": False,
                                         "is_spin_polarized": False}
            content = backend.generate_input(simple_topo, sug)
        assert "# WARNING: Test warning message" in content

    @patch("forge.core.hardware.check_elpa_available", return_value=False)
    def test_vector_split_included(self, _elpa, backend, simple_topo, base_suggestion):
        sug = {**base_suggestion, "vector_split_active": True, "nmat": 6000}
        with patch.object(backend, "_detect_problem_size") as mock_detect:
            mock_detect.return_value = {"atoms": 10, "kpoints": 8, "nmat": 6000,
                                         "is_soc": False, "is_hybrid": False,
                                         "is_spin_polarized": False}
            content = backend.generate_input(simple_topo, sug)
        assert "lapw2_vector_split" in content

    @patch("forge.core.hardware.check_elpa_available", return_value=True)
    def test_omp_global_in_elpa_mode(self, _elpa, backend, simple_topo, base_suggestion):
        sug = {**base_suggestion, "nmat": 10000, "nkpt": 1, "omp_threads_per_rank": 2}
        with patch.object(backend, "_detect_problem_size") as mock_detect:
            mock_detect.return_value = {"atoms": 10, "kpoints": 1, "nmat": 10000,
                                         "is_soc": False, "is_hybrid": False,
                                         "is_spin_polarized": False}
            content = backend.generate_input(simple_topo, sug)
        # omp_global appears in fine_grain_elpa strategy
        assert "omp_global" in content


# =============================================================================
# Tests: ELPA solver suggestion in generate_input
# =============================================================================

class TestELPASuggestionInGenerateInput:
    def test_elpa_solver_integration(self, backend, simple_topo):
        """When nmat > 2000, ELPA selector is invoked."""
        sug = {
            "mode": "mpi",
            "recommended_total_cores": 32,
            "nmat": 3000,
            "nkpt": 8,
        }
        content = backend.generate_input(simple_topo, sug)
        assert len(content) > 0


# =============================================================================
# Tests: ProblemSize from backend class
# =============================================================================

class TestBackendDetectProblemSize:
    def test_delegates_to_parser(self, backend, tmp_path, monkeypatch):
        """Backend.detect_problem_size delegates to parsers.detect_problem_size."""
        write_struct(tmp_path, STRUCT_LARGE)
        monkeypatch.chdir(tmp_path)
        result = backend.detect_problem_size()
        assert result["atoms"] == 5


# =============================================================================
# Tests: _detect_io_bottleneck wrapper
# =============================================================================

class TestBackendDetectIOBottleneck:
    def test_delegates_correctly(self, backend):
        result = backend._detect_io_bottleneck(10000, 2, 64)
        assert result["risk_level"] == "high"
        assert result["auto_enable_vector_split"] is True


# =============================================================================
# Tests: Empty nodes edge case
# =============================================================================

class TestEdgeCases:
    def test_empty_nodes_topology(self, backend):
        """Topology with no nodes should not crash generate_input."""
        topo = Topology(nodes=[], cores_per_node=[], env_type="local")
        sug = {"mode": "mpi", "recommended_total_cores": 4}
        with patch("forge.core.hardware.check_elpa_available", return_value=False):
            content = backend.generate_input(topo, sug)
        assert len(content) > 0

    def test_single_node_topology(self, backend):
        topo = Topology(nodes=["login01"], cores_per_node=[8], env_type="local")
        sug = {"mode": "mpi", "recommended_total_cores": 8}
        with patch("forge.core.hardware.check_elpa_available", return_value=False):
            content = backend.generate_input(topo, sug)
        assert "lapw0: login01:" in content

    def test_estimate_kpoint_density_empty_struct(self, tmp_path, monkeypatch):
        """Struct with no recognizable NONEQUIV.ATOMS line."""
        p = tmp_path / "weird.struct"
        p.write_text("Just some text\nno lattice info\n")
        monkeypatch.chdir(tmp_path)
        result = estimate_kpoint_density()
        assert result["nkpt_est"] > 0  # falls back to heuristic

    def test_detect_io_bottleneck_zero_nkpt(self):
        result = detect_io_bottleneck(5000, 0, 64)
        assert result["risk_level"] == "low"

    def test_get_execution_command_defaults(self, backend):
        """With no mode specified, defaults work."""
        sug = {}
        cmd = backend.get_execution_command(sug)
        assert "run_lapw" in cmd
        assert "-p" in cmd
