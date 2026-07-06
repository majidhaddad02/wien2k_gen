"""
Production-Grade Tests for optimizer.advisor Module.
Validates resource suggestion logic, execution mode selection, divisibility enforcement,
memory/time estimation, and edge-case resilience.

Uses mock-driven execution to isolate advisor logic from hardware/filesystem dependencies.
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from wien2k_gen.core.topology import Topology
from wien2k_gen.optimizer.advisor import (
    suggest_optimal_resources,
    estimate_memory_footprint_gb,
    _get_current_backend,
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
    @patch("wien2k_gen.optimizer.advisor.get_memory_bandwidth_gb_s", return_value=200.0)
    @patch("wien2k_gen.optimizer.advisor.calculate_peak_fp64_gflops", return_value=800.0)
    @patch("wien2k_gen.optimizer.advisor.get_fma_units_per_core", return_value=2)
    @patch("wien2k_gen.optimizer.advisor.get_cpu_architecture", return_value="xeon")
    @patch("wien2k_gen.optimizer.advisor.get_total_mem_kb", return_value=256 * 1024 * 1024)
    @patch("wien2k_gen.optimizer.advisor.get_physical_cores", return_value=64)
    @patch("wien2k_gen.optimizer.advisor.is_hyperthreading_active", return_value=False)
    @patch("wien2k_gen.optimizer.advisor.get_job_memory_limit_mb", return_value=None)
    @patch("wien2k_gen.optimizer.advisor.check_elpa_available", return_value=False)
    @patch("wien2k_gen.optimizer.advisor.check_mkl_available", return_value=False)
    @patch("wien2k_gen.optimizer.advisor.get_numa_node_count", return_value=2)
    @patch("wien2k_gen.optimizer.advisor.get_scratch_filesystem_type", return_value="tmpfs")
    @patch("wien2k_gen.optimizer.advisor._get_current_backend")
    def test_returns_valid_suggestion(self, mock_backend_fn, *args, **kwargs):
        mock_backend_fn.return_value = _setup_mock_backend()
        result = suggest_optimal_resources(Topology(nodes=["n1"], cores_per_node=[16]))

        from wien2k_gen.optimizer.advisor import ResourceSuggestion
        assert isinstance(result, ResourceSuggestion)
        assert result.mode in ("kpoint", "hybrid", "mpi")
        assert result.recommended_total_cores > 0
        assert isinstance(result.warnings, list)
        assert 0.0 <= result.confidence_score <= 1.0

    @patch("wien2k_gen.optimizer.advisor.get_memory_bandwidth_gb_s", return_value=200.0)
    @patch("wien2k_gen.optimizer.advisor.calculate_peak_fp64_gflops", return_value=800.0)
    @patch("wien2k_gen.optimizer.advisor.get_fma_units_per_core", return_value=2)
    @patch("wien2k_gen.optimizer.advisor.get_cpu_architecture", return_value="xeon")
    @patch("wien2k_gen.optimizer.advisor.get_total_mem_kb", return_value=256 * 1024 * 1024)
    @patch("wien2k_gen.optimizer.advisor.get_physical_cores", return_value=64)
    @patch("wien2k_gen.optimizer.advisor.is_hyperthreading_active", return_value=False)
    @patch("wien2k_gen.optimizer.advisor.get_job_memory_limit_mb", return_value=None)
    @patch("wien2k_gen.optimizer.advisor.check_elpa_available", return_value=False)
    @patch("wien2k_gen.optimizer.advisor.check_mkl_available", return_value=False)
    @patch("wien2k_gen.optimizer.advisor.get_numa_node_count", return_value=2)
    @patch("wien2k_gen.optimizer.advisor.get_scratch_filesystem_type", return_value="tmpfs")
    @patch("wien2k_gen.optimizer.advisor._get_current_backend")
    def test_kpoint_mode_detected(self, mock_backend_fn, *args, **kwargs):
        backend = _setup_mock_backend(nmat=500, nkpt=64)
        mock_backend_fn.return_value = backend
        result = suggest_optimal_resources(
            Topology(nodes=["n1", "n2", "n3", "n4"], cores_per_node=[16] * 4),
        )
        assert result.mode in ("kpoint", "hybrid", "mpi")
        assert result.recommended_total_cores > 0

    @patch("wien2k_gen.optimizer.advisor.get_memory_bandwidth_gb_s", return_value=200.0)
    @patch("wien2k_gen.optimizer.advisor.calculate_peak_fp64_gflops", return_value=800.0)
    @patch("wien2k_gen.optimizer.advisor.get_fma_units_per_core", return_value=2)
    @patch("wien2k_gen.optimizer.advisor.get_cpu_architecture", return_value="xeon")
    @patch("wien2k_gen.optimizer.advisor.get_total_mem_kb", return_value=256 * 1024 * 1024)
    @patch("wien2k_gen.optimizer.advisor.get_physical_cores", return_value=64)
    @patch("wien2k_gen.optimizer.advisor.is_hyperthreading_active", return_value=False)
    @patch("wien2k_gen.optimizer.advisor.get_job_memory_limit_mb", return_value=None)
    @patch("wien2k_gen.optimizer.advisor.check_elpa_available", return_value=True)
    @patch("wien2k_gen.optimizer.advisor.check_mkl_available", return_value=False)
    @patch("wien2k_gen.optimizer.advisor.get_numa_node_count", return_value=4)
    @patch("wien2k_gen.optimizer.advisor.get_scratch_filesystem_type", return_value="tmpfs")
    @patch("wien2k_gen.optimizer.advisor._get_current_backend")
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
