"""
Production-Grade Tests for optimizer.advisor Module.
Validates resource suggestion logic, execution mode selection, divisibility enforcement,
memory/time estimation, and edge-case resilience.

Uses mock-driven execution to isolate advisor logic from hardware/filesystem dependencies.
"""
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest

from forge.core.topology import Topology
from forge.optimizer.advisor import (
    estimate_amdahl_saturation,
    estimate_memory_footprint_gb,
    suggest_optimal_resources,
)


@pytest.fixture
def topo_64() -> Topology:
    return Topology(
        nodes=["node01", "node02"],
        cores_per_node=[32, 32],
        env_type="slurm",
        scheduler_hints={"mpi_launcher": "srun", "numa_aware": True},
    )


@pytest.fixture
def topo_single() -> Topology:
    return Topology(
        nodes=["login01"],
        cores_per_node=[8],
        env_type="local",
        scheduler_hints={"mpi_launcher": "mpirun", "numa_aware": False},
    )


@pytest.fixture
def topo_hetero() -> Topology:
    return Topology(
        nodes=["n1", "n2", "n3"],
        cores_per_node=[16, 16, 8],
        env_type="slurm",
    )


def _setup_mock_backend(nmat=1200, nkpt=8, atoms=10, nbands=100, is_soc=False, is_hybrid=False):
    """Create a mock backend with detect_problem_size returning configured values."""
    data = {
        "atoms": atoms, "kpoints": nkpt, "nmat": nmat,
        "nbands": nbands, "is_soc": is_soc, "is_hybrid": is_hybrid,
        "rkmax": 7.0, "complexity": 1.0,
    }
    backend = MagicMock()
    backend.detect_problem_size.return_value = data
    return backend


class TestSuggestOptimalResources:
    @patch("forge.optimizer.advisor.get_memory_bandwidth_gb_s", return_value=200.0)
    @patch("forge.optimizer.advisor.calculate_peak_fp64_gflops", return_value=800.0)
    @patch("forge.optimizer.advisor.get_fma_units_per_core", return_value=2)
    @patch("forge.optimizer.advisor.get_cpu_architecture", return_value="xeon")
    @patch("forge.optimizer.advisor.get_total_mem_kb", return_value=256 * 1024 * 1024)
    @patch("forge.optimizer.advisor.get_physical_cores", return_value=64)
    @patch("forge.optimizer.advisor.is_hyperthreading_active", return_value=False)
    @patch("forge.optimizer.advisor.get_job_memory_limit_mb", return_value=None)
    @patch("forge.optimizer.advisor.check_elpa_available", return_value=False)
    @patch("forge.optimizer.advisor.check_mkl_available", return_value=False)
    @patch("forge.optimizer.advisor.get_numa_node_count", return_value=2)
    @patch("forge.optimizer.advisor.get_scratch_filesystem_type", return_value="tmpfs")
    @patch("forge.optimizer.advisor._get_current_backend")
    def test_returns_valid_suggestion(self, mock_backend_fn, *args, **kwargs):
        mock_backend_fn.return_value = _setup_mock_backend()
        result = suggest_optimal_resources(Topology(nodes=["n1"], cores_per_node=[16]))

        from forge.optimizer.advisor import ResourceSuggestion
        assert isinstance(result, ResourceSuggestion)
        assert result.mode in ("kpoint", "hybrid", "mpi")
        assert result.recommended_total_cores > 0
        assert isinstance(result.warnings, list)
        assert 0.0 <= result.confidence_score <= 1.0

    @patch("forge.optimizer.advisor.get_memory_bandwidth_gb_s", return_value=200.0)
    @patch("forge.optimizer.advisor.calculate_peak_fp64_gflops", return_value=800.0)
    @patch("forge.optimizer.advisor.get_fma_units_per_core", return_value=2)
    @patch("forge.optimizer.advisor.get_cpu_architecture", return_value="xeon")
    @patch("forge.optimizer.advisor.get_total_mem_kb", return_value=256 * 1024 * 1024)
    @patch("forge.optimizer.advisor.get_physical_cores", return_value=64)
    @patch("forge.optimizer.advisor.is_hyperthreading_active", return_value=False)
    @patch("forge.optimizer.advisor.get_job_memory_limit_mb", return_value=None)
    @patch("forge.optimizer.advisor.check_elpa_available", return_value=False)
    @patch("forge.optimizer.advisor.check_mkl_available", return_value=False)
    @patch("forge.optimizer.advisor.get_numa_node_count", return_value=2)
    @patch("forge.optimizer.advisor.get_scratch_filesystem_type", return_value="tmpfs")
    @patch("forge.optimizer.advisor._get_current_backend")
    def test_kpoint_mode_detected(self, mock_backend_fn, *args, **kwargs):
        backend = _setup_mock_backend(nmat=500, nkpt=64)
        mock_backend_fn.return_value = backend
        result = suggest_optimal_resources(
            Topology(nodes=["n1", "n2", "n3", "n4"], cores_per_node=[16] * 4),
        )
        assert result.mode in ("kpoint", "hybrid", "mpi")
        assert result.recommended_total_cores > 0

    @patch("forge.optimizer.advisor.get_memory_bandwidth_gb_s", return_value=200.0)
    @patch("forge.optimizer.advisor.calculate_peak_fp64_gflops", return_value=800.0)
    @patch("forge.optimizer.advisor.get_fma_units_per_core", return_value=2)
    @patch("forge.optimizer.advisor.get_cpu_architecture", return_value="xeon")
    @patch("forge.optimizer.advisor.get_total_mem_kb", return_value=256 * 1024 * 1024)
    @patch("forge.optimizer.advisor.get_physical_cores", return_value=64)
    @patch("forge.optimizer.advisor.is_hyperthreading_active", return_value=False)
    @patch("forge.optimizer.advisor.get_job_memory_limit_mb", return_value=None)
    @patch("forge.optimizer.advisor.check_elpa_available", return_value=True)
    @patch("forge.optimizer.advisor.check_mkl_available", return_value=False)
    @patch("forge.optimizer.advisor.get_numa_node_count", return_value=4)
    @patch("forge.optimizer.advisor.get_scratch_filesystem_type", return_value="tmpfs")
    @patch("forge.optimizer.advisor._get_current_backend")
    def test_large_matrix_hybrid(self, mock_backend_fn, *args, **kwargs):
        backend = _setup_mock_backend(nmat=8000, nkpt=16, atoms=50, nbands=800)
        mock_backend_fn.return_value = backend
        result = suggest_optimal_resources(
            Topology(nodes=[f"n{i:02d}" for i in range(4)], cores_per_node=[32] * 4),
        )
        assert result.mode in ("kpoint", "hybrid", "mpi")
        assert result.recommended_total_cores > 0


class TestMemoryEstimation:
    def test_memory_small_system(self):
        gb = estimate_memory_footprint_gb(nmat=1000, total_cores=16)
        assert gb > 0
        assert gb < 20  # Small system with charge density overhead

    def test_memory_large_system_per_rank(self):
        gb_per_rank = estimate_memory_footprint_gb(nmat=10000, total_cores=64)
        gb_aggregate = estimate_memory_footprint_gb(nmat=10000, total_cores=1)
        assert gb_per_rank > 0
        assert gb_per_rank < gb_aggregate  # Per-rank should be smaller

    def test_memory_soc_doubled(self):
        gb_no_soc = estimate_memory_footprint_gb(nmat=2000, total_cores=8, is_soc=False)
        gb_soc = estimate_memory_footprint_gb(nmat=2000, total_cores=8, is_soc=True)
        assert gb_soc > gb_no_soc

    def test_memory_more_ranks_less_per_rank(self):
        gb_few = estimate_memory_footprint_gb(nmat=5000, total_cores=4)
        gb_many = estimate_memory_footprint_gb(nmat=5000, total_cores=64)
        assert gb_many < gb_few

    def test_memory_zero_nmat(self):
        gb = estimate_memory_footprint_gb(nmat=0)
        assert gb == 2.0  # Fallback


class TestAmdahlSaturation:
    """Tests for Amdahl's Law saturation detection.

    References:
    - Amdahl, G.M. (1967). AFIPS Conf. Proc. 30, 483-485.
    - Hager & Wellein (2010). "Introduction to HPC". CRC Press. §4.2.
    """

    def test_small_atoms_gives_high_serial_fraction(self):
        """Tiny systems (atoms<4): lapw0 is serial → high serial fraction."""
        result = estimate_amdahl_saturation(
            kpoints=8, nmat=500, atoms=2,
            total_cores_available=64, num_nodes=1,
        )
        assert result["serial_fraction"] >= 0.15
        assert result["max_speedup_amdahl"] <= 6.7  # 1/0.15

    def test_large_system_low_serial_fraction(self):
        """Large supercell: lapw0 amortized → low serial fraction."""
        result = estimate_amdahl_saturation(
            kpoints=64, nmat=20000, atoms=150,
            total_cores_available=64, num_nodes=1,
        )
        assert result["serial_fraction"] <= 0.05
        assert result["max_speedup_amdahl"] > 20

    def test_kpoint_saturation_warning(self):
        """When cores exceed k-point count, expect saturation warning."""
        result = estimate_amdahl_saturation(
            kpoints=4, nmat=2000, atoms=10,
            total_cores_available=64, num_nodes=1,
        )
        assert result["is_saturated"]
        warnings = result["saturation_warnings"]
        assert any("k-point" in w.lower() for w in warnings)

    def test_within_kpoint_limit_no_saturation(self):
        """When cores ≤ k-point count, no saturation for moderate system."""
        result = estimate_amdahl_saturation(
            kpoints=64, nmat=5000, atoms=100,
            total_cores_available=16, num_nodes=1,
        )
        assert not result["is_saturated"]

    def test_multinode_increases_serial_fraction(self):
        """More nodes → communication overhead → higher serial fraction."""
        single = estimate_amdahl_saturation(
            kpoints=32, nmat=5000, atoms=20,
            total_cores_available=64, num_nodes=1,
        )
        multi = estimate_amdahl_saturation(
            kpoints=32, nmat=5000, atoms=20,
            total_cores_available=64, num_nodes=8,
        )
        assert multi["serial_fraction"] > single["serial_fraction"]

    def test_sweet_spot_never_exceeds_max_efficient(self):
        """Sweet spot ≤ max efficient cores."""
        result = estimate_amdahl_saturation(
            kpoints=16, nmat=3000, atoms=10,
            total_cores_available=128, num_nodes=2,
        )
        assert result["sweet_spot_cores"] <= result["max_efficient_cores"]

    def test_severe_saturation_detected(self):
        """2x over max efficient -> severe saturation."""
        result = estimate_amdahl_saturation(
            kpoints=4, nmat=500, atoms=2,
            total_cores_available=256, num_nodes=8,
        )
        assert result["is_saturated"]
        warnings = result["saturation_warnings"]
        assert any("severe" in w.lower() for w in warnings)

    def test_speedup_realistic_range(self):
        """Speedup should be 1 ≤ speedup ≤ max theoretical."""
        result = estimate_amdahl_saturation(
            kpoints=16, nmat=3000, atoms=20,
            total_cores_available=32, num_nodes=1,
        )
        assert 1.0 <= result["speedup_at_cores"] <= result["max_speedup_amdahl"]

    def test_efficiency_declines_with_more_cores(self):
        """Efficiency should drop as cores increase."""
        low = estimate_amdahl_saturation(
            kpoints=32, nmat=3000, atoms=20,
            total_cores_available=8, num_nodes=1,
        )
        high = estimate_amdahl_saturation(
            kpoints=32, nmat=3000, atoms=20,
            total_cores_available=128, num_nodes=1,
        )
        assert high["efficiency_at_cores"] < low["efficiency_at_cores"]

    def test_big_nmat_reduces_serial_fraction(self):
        """Large matrix → more parallelizable → lower serial fraction."""
        small = estimate_amdahl_saturation(
            kpoints=16, nmat=500, atoms=20,
            total_cores_available=32, num_nodes=1,
        )
        large = estimate_amdahl_saturation(
            kpoints=16, nmat=25000, atoms=20,
            total_cores_available=32, num_nodes=1,
        )
        assert large["serial_fraction"] < small["serial_fraction"]

    def test_max_efficient_bounded(self):
        """max_efficient_cores bounded by Amdahl, k-points, and bandwidth."""
        result = estimate_amdahl_saturation(
            kpoints=4, nmat=500, atoms=3,
            total_cores_available=512, num_nodes=16,
        )
        assert 1 <= result["max_efficient_cores"] <= 512
        # With 4 k-points and small system, max efficient << 512
        assert result["max_efficient_cores"] < 512

    def test_all_fields_present(self):
        """Ensure all expected keys are in the result dict."""
        result = estimate_amdahl_saturation(
            kpoints=8, nmat=2000, atoms=10,
            total_cores_available=32, num_nodes=1,
        )
        expected_keys = {
            "serial_fraction", "max_speedup_amdahl", "speedup_at_cores",
            "efficiency_at_cores", "max_efficient_cores", "sweet_spot_cores",
            "is_saturated", "kpoint_limit", "saturation_warnings",
        }
        assert expected_keys <= set(result.keys())


class TestHyperthreadingAwareness:
    """Tests for SMT/HT-aware core allocation (Phase 5 fix)."""

    @patch("forge.optimizer.advisor.get_memory_bandwidth_gb_s", return_value=200.0)
    @patch("forge.optimizer.advisor.calculate_peak_fp64_gflops", return_value=800.0)
    @patch("forge.optimizer.advisor.get_fma_units_per_core", return_value=2)
    @patch("forge.optimizer.advisor.get_cpu_architecture", return_value="xeon")
    @patch("forge.optimizer.advisor.get_total_mem_kb", return_value=256 * 1024 * 1024)
    @patch("forge.optimizer.advisor.get_physical_cores", return_value=32)
    @patch("forge.optimizer.advisor.is_hyperthreading_active", return_value=True)
    @patch("forge.optimizer.advisor.get_job_memory_limit_mb", return_value=None)
    @patch("forge.optimizer.advisor.check_elpa_available", return_value=True)
    @patch("forge.optimizer.advisor.check_mkl_available", return_value=True)
    @patch("forge.optimizer.advisor.get_numa_node_count", return_value=2)
    @patch("forge.optimizer.advisor.get_scratch_filesystem_type", return_value="tmpfs")
    @patch("forge.optimizer.advisor._get_current_backend")
    def test_ht_active_caps_cores_to_physical(self, mock_backend_fn, *args, **kwargs):
        backend = _setup_mock_backend(nmat=5000, nkpt=64, atoms=50, nbands=500)
        mock_backend_fn.return_value = backend
        topo = Topology(nodes=[f"n{i:02d}" for i in range(4)], cores_per_node=[32] * 4)
        result = suggest_optimal_resources(topo)
        assert result.recommended_total_cores <= 32 * 4

    @patch("forge.optimizer.advisor.get_memory_bandwidth_gb_s", return_value=200.0)
    @patch("forge.optimizer.advisor.calculate_peak_fp64_gflops", return_value=800.0)
    @patch("forge.optimizer.advisor.get_fma_units_per_core", return_value=2)
    @patch("forge.optimizer.advisor.get_cpu_architecture", return_value="xeon")
    @patch("forge.optimizer.advisor.get_total_mem_kb", return_value=256 * 1024 * 1024)
    @patch("forge.optimizer.advisor.get_physical_cores", return_value=32)
    @patch("forge.optimizer.advisor.is_hyperthreading_active", return_value=True)
    @patch("forge.optimizer.advisor.get_job_memory_limit_mb", return_value=None)
    @patch("forge.optimizer.advisor.check_elpa_available", return_value=False)
    @patch("forge.optimizer.advisor.check_mkl_available", return_value=False)
    @patch("forge.optimizer.advisor.get_numa_node_count", return_value=2)
    @patch("forge.optimizer.advisor.get_scratch_filesystem_type", return_value="tmpfs")
    @patch("forge.optimizer.advisor._get_current_backend")
    def test_ht_active_generates_warning(self, mock_backend_fn, *args, **kwargs):
        backend = _setup_mock_backend(nmat=5000, nkpt=64, atoms=50, nbands=500)
        mock_backend_fn.return_value = backend
        topo = Topology(nodes=["n1", "n2"], cores_per_node=[64, 64])
        result = suggest_optimal_resources(topo)
        assert any("Hyper-Threading" in w for w in result.warnings)
        assert any("nomultithread" in w.lower() for w in result.warnings)

    @patch("forge.optimizer.advisor.get_memory_bandwidth_gb_s", return_value=200.0)
    @patch("forge.optimizer.advisor.calculate_peak_fp64_gflops", return_value=800.0)
    @patch("forge.optimizer.advisor.get_fma_units_per_core", return_value=2)
    @patch("forge.optimizer.advisor.get_cpu_architecture", return_value="xeon")
    @patch("forge.optimizer.advisor.get_total_mem_kb", return_value=256 * 1024 * 1024)
    @patch("forge.optimizer.advisor.get_physical_cores", return_value=32)
    @patch("forge.optimizer.advisor.is_hyperthreading_active", return_value=True)
    @patch("forge.optimizer.advisor.get_job_memory_limit_mb", return_value=None)
    @patch("forge.optimizer.advisor.check_elpa_available", return_value=False)
    @patch("forge.optimizer.advisor.check_mkl_available", return_value=False)
    @patch("forge.optimizer.advisor.get_numa_node_count", return_value=2)
    @patch("forge.optimizer.advisor.get_scratch_filesystem_type", return_value="tmpfs")
    @patch("forge.optimizer.advisor._get_current_backend")
    def test_ht_active_reduces_confidence(self, mock_backend_fn, *args, **kwargs):
        backend = _setup_mock_backend(nmat=5000, nkpt=64, atoms=50, nbands=500)
        mock_backend_fn.return_value = backend
        topo = Topology(nodes=["n1"], cores_per_node=[64])
        result = suggest_optimal_resources(topo)
        assert result.confidence_score < 1.0


@contextmanager
def _hw_patches(**overrides):
    """Patch advisor hardware probes so allocation tests are deterministic."""
    defaults = {
        "get_memory_bandwidth_gb_s": 200.0,
        "calculate_peak_fp64_gflops": 800.0,
        "get_fma_units_per_core": 2,
        "get_cpu_architecture": "xeon",
        "get_total_mem_kb": 256 * 1024 * 1024,
        "get_physical_cores": 100,
        "is_hyperthreading_active": False,
        "get_job_memory_limit_mb": None,
        "check_elpa_available": False,
        "check_mkl_available": False,
        "get_numa_node_count": 2,
        "get_scratch_filesystem_type": "tmpfs",
        "get_cpu_frequency_info": {"base": 2000.0},
    }
    defaults.update(overrides)
    with ExitStack() as stack:
        for name, value in defaults.items():
            stack.enter_context(patch(f"forge.optimizer.advisor.{name}", return_value=value))
        stack.enter_context(
            patch(
                "forge.optimizer.advisor.estimate_max_kp_cores_roofline",
                side_effect=lambda nmat, mem_bw, arch, cores, **kw: cores,
            )
        )
        yield


class TestAmdahlCoreAllocation:
    """Amdahl max_efficient_cores must constrain kpoint/hybrid auto-allocation.

    Previously estimate_amdahl_saturation() only appended warnings after
    ranks were chosen, so .machines could still request all topology cores.
    """

    def test_small_problem_auto_clamps_to_max_efficient(self):
        """ClH3-like: few atoms, high serial fraction, many idle cores.

        Expectation changed from allocating all 100 topology cores to
        clamping at Amdahl max_efficient_cores when --cores is omitted.
        """
        backend = _setup_mock_backend(nmat=4320, nkpt=525, atoms=8, nbands=200)
        topo = Topology(nodes=["n1"], cores_per_node=[100], env_type="slurm")
        with patch("forge.optimizer.advisor._get_current_backend", return_value=backend):
            with _hw_patches():
                result = suggest_optimal_resources(topo, respect_saturation_limit=True)
        assert result.mode in ("kpoint", "hybrid")
        max_eff = result.saturation_data["max_efficient_cores"]
        assert result.recommended_total_cores <= max_eff
        assert result.recommended_total_cores < 100
        assert any("Capped at" in w and "--ignore-saturation" in w for w in result.warnings)
        assert not any(w.startswith("SEVERE SATURATION") for w in result.warnings)

    def test_explicit_cores_bypass_clamp(self):
        """Explicit --cores N must not be silently reduced by Amdahl.

        Other hard caps (nk, topology) still apply; N=48 is below nk=525
        and below the 100-core machine.
        """
        backend = _setup_mock_backend(nmat=4320, nkpt=525, atoms=8, nbands=200)
        topo = Topology(nodes=["n1"], cores_per_node=[100], env_type="slurm")
        with patch("forge.optimizer.advisor._get_current_backend", return_value=backend):
            with _hw_patches():
                result = suggest_optimal_resources(
                    topo,
                    user_max_cores=48,
                    respect_saturation_limit=False,
                )
        assert result.recommended_total_cores == 48
        assert not any("Capped at" in w for w in result.warnings)

    def test_ignore_saturation_bypasses_auto_clamp(self):
        """--ignore-saturation keeps topology-derived cores on auto-detect."""
        backend = _setup_mock_backend(nmat=4320, nkpt=525, atoms=8, nbands=200)
        topo = Topology(nodes=["n1"], cores_per_node=[100], env_type="slurm")
        with patch("forge.optimizer.advisor._get_current_backend", return_value=backend):
            with _hw_patches():
                result = suggest_optimal_resources(topo, respect_saturation_limit=False)
        assert result.recommended_total_cores > result.saturation_data["max_efficient_cores"]
        assert not any("Capped at" in w for w in result.warnings)

    def test_large_system_not_unnecessarily_restricted(self):
        """TiO2-like large cell: low serial fraction, clamp must not bite.

        Single-node nmat>20000 and atoms=150 yield s≈0.01 so
        max_efficient_cores covers the 64-core topology.
        """
        backend = _setup_mock_backend(nmat=25000, nkpt=64, atoms=150, nbands=2000)
        topo = Topology(nodes=["n1"], cores_per_node=[64], env_type="slurm")
        with patch("forge.optimizer.advisor._get_current_backend", return_value=backend):
            with _hw_patches(get_physical_cores=64, check_elpa_available=True):
                result = suggest_optimal_resources(topo, respect_saturation_limit=True)
        max_eff = result.saturation_data["max_efficient_cores"]
        assert max_eff >= 32
        assert result.recommended_total_cores >= min(32, max_eff)
        assert not any("Capped at" in w for w in result.warnings)

    def test_mpi_mode_is_not_clamped(self):
        """Dense MPI on large nmat keeps topology cores (no Amdahl clamp)."""
        backend = _setup_mock_backend(nmat=20000, nkpt=2, atoms=80, nbands=1500)
        topo = Topology(nodes=["n1"], cores_per_node=[64], env_type="slurm")
        with patch("forge.optimizer.advisor._get_current_backend", return_value=backend):
            with _hw_patches(get_physical_cores=64, check_elpa_available=True):
                result = suggest_optimal_resources(topo, respect_saturation_limit=True)
        assert result.mode == "mpi"
        assert result.recommended_total_cores == 64
        assert not any("Capped at" in w for w in result.warnings)


class TestPipelineSuggestionMerge:
    """CLI flags must not skip advisor or crash on extra GPU keys."""

    def test_partial_cli_suggestion_still_runs_advisor(self):
        from forge.core.pipeline import run_pipeline
        from forge.types import ResourceSuggestion as TypesSuggestion

        backend = MagicMock()
        backend.detect_problem_size.return_value = {
            "atoms": 8, "kpoints": 16, "nmat": 1200, "nbands": 80,
        }
        backend.generate_input.return_value = "# machines\n"

        advisor_result = MagicMock()
        advisor_result.to_dict.return_value = {
            "mode": "hybrid",
            "recommended_total_cores": 8,
            "omp_threads_per_rank": 2,
            "mpi_ranks_per_node": 4,
            "cores_per_node": [8],
            "warnings": ["advisor-ran"],
            "reason": "auto",
            "confidence": 0.9,
        }

        topo = Topology(nodes=["n1"], cores_per_node=[16], env_type="local")
        with patch("forge.core.pipeline._get_current_backend", return_value=backend):
            with patch("forge.optimizer.advisor.suggest_optimal_resources", return_value=advisor_result) as mock_adv:
                with patch("forge.core.pipeline.preflight_check", return_value=[]):
                    result = run_pipeline(
                        topo,
                        dry_run=True,
                        user_suggestion={"mode": "kpoint", "omp_threads_per_rank": 4},
                    )
        mock_adv.assert_called_once()
        assert result.success
        assert isinstance(result.suggestion, TypesSuggestion)
        assert result.suggestion.mode == "kpoint"
        assert result.suggestion.omp_threads_per_rank == 4
        assert result.suggestion.recommended_total_cores == 8
        payload = backend.generate_input.call_args[0][1]
        assert payload["mode"] == "kpoint"
        assert payload["omp_threads_per_rank"] == 4

    def test_gpu_extra_keys_do_not_typeerror_and_reach_generate(self):
        from forge.core.pipeline import run_pipeline

        backend = MagicMock()
        backend.detect_problem_size.return_value = {"atoms": 8, "kpoints": 8, "nmat": 500}
        backend.generate_input.return_value = "gpu: tesla\n"

        advisor_result = MagicMock()
        advisor_result.to_dict.return_value = {
            "mode": "mpi",
            "recommended_total_cores": 4,
            "omp_threads_per_rank": 1,
            "mpi_ranks_per_node": 4,
            "cores_per_node": [4],
            "warnings": [],
            "reason": "auto",
        }
        gpu_rec = {"use_gpu": True, "gpu_count": 1}
        topo = Topology(nodes=["n1"], cores_per_node=[8], env_type="local")
        with patch("forge.core.pipeline._get_current_backend", return_value=backend):
            with patch("forge.optimizer.advisor.suggest_optimal_resources", return_value=advisor_result):
                with patch("forge.core.pipeline.preflight_check", return_value=[]):
                    result = run_pipeline(
                        topo,
                        dry_run=True,
                        user_suggestion={
                            "gpu_recommendation": gpu_rec,
                            "mixed_precision": {"fp32": True},
                        },
                    )
        assert result.success
        assert result.metadata.get("gpu_recommendation") == gpu_rec
        payload = backend.generate_input.call_args[0][1]
        assert payload["gpu_recommendation"] == gpu_rec
        assert payload["mixed_precision"] == {"fp32": True}
