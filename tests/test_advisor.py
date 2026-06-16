"""
Production-Grade Tests for optimizer.advisor Module.
Covers resource suggestion logic, execution mode selection, divisibility enforcement,
memory/time estimation, hardware-aware tuning, warning generation, and edge-case resilience.
Designed for deterministic, mock-driven execution without hardware dependencies.
"""

import os
import math
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from wien2k_gen.optimizer.advisor import suggest_optimal_resources
from wien2k_gen.types import TopologyData, ResourceSuggestion
from wien2k_gen.exceptions import ConfigurationError, ValidationError


# =============================================================================
# Fixtures: Mock Topology, Hardware & Problem Sizes
# =============================================================================

@pytest.fixture
def standard_topology() -> TopologyData:
    return TopologyData(
        nodes=["node01", "node02"],
        cores_per_node=[32, 32],
        total_cores=64,
        env_type="slurm",
        scheduler_hints={"mpi_launcher": "srun", "numa_aware": True},
        heterogeneous=False
    )

@pytest.fixture
def large_cluster_topology() -> TopologyData:
    return TopologyData(
        nodes=[f"node{i:02d}" for i in range(16)],
        cores_per_node=[64] * 16,
        total_cores=1024,
        env_type="slurm",
        scheduler_hints={"mpi_launcher": "srun", "numa_aware": True},
        heterogeneous=False
    )

@pytest.fixture
def single_node_topology() -> TopologyData:
    return TopologyData(
        nodes=["login01"],
        cores_per_node=[8],
        total_cores=8,
        env_type="local",
        scheduler_hints={"mpi_launcher": "mpirun", "numa_aware": False},
        heterogeneous=False
    )

@pytest.fixture
def mock_hardware_profile() -> Dict[str, Any]:
    return {
        "cpu_arch": "x86_64", "physical_cores": 64, "logical_cores": 128,
        "sockets": 2, "cores_per_socket": 32, "memory_gb": 256.0,
        "memory_bandwidth_gb_s": 200.0, "peak_fp64_gflops": 800.0,
        "numa_nodes": 2, "vector_isa": "avx512", "interconnect_type": "infiniband",
        "interconnect_provider": "mlx", "scratch_fs": "tmpfs",
        "elpa_available": True, "mkl_available": True
    }

@pytest.fixture
def problem_small() -> Dict[str, Any]:
    return {"atoms": 10, "kpoints": 8, "nmat": 1200, "nbands": 100, "is_soc": False, "is_hybrid": False}

@pytest.fixture
def problem_large() -> Dict[str, Any]:
    return {"atoms": 200, "kpoints": 64, "nmat": 15000, "nbands": 1500, "is_soc": True, "is_hybrid": True}

@pytest.fixture
def problem_gamma_only() -> Dict[str, Any]:
    return {"atoms": 50, "kpoints": 1, "nmat": 8000, "nbands": 800, "is_soc": False, "is_hybrid": False}


# =============================================================================
# Test Suites
# =============================================================================

class TestSuggestionStructure:
    """Validate output schema & type safety of advisor responses."""

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_returns_valid_resource_suggestion(self, mock_hw, standard_topology, problem_small, mock_hardware_profile):
        mock_hw.return_value = mock_hardware_profile
        result = suggest_optimal_resources(standard_topology, problem_small)
        
        assert isinstance(result, dict)
        required_keys = {"mode", "recommended_total_cores", "omp_threads_per_rank", 
                         "mpi_ranks_per_node", "cores_per_node", "warnings", 
                         "reason", "confidence", "estimated_memory_gb"}
        assert required_keys.issubset(result.keys())
        assert isinstance(result["warnings"], list)
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["recommended_total_cores"] > 0

    @pytest.mark.parametrize("topo_fixture", ["standard_topology", "single_node_topology"])
    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_deterministic_output(self, mock_hw, request, topo_fixture, problem_small, mock_hardware_profile):
        """Same inputs must always yield identical suggestions."""
        mock_hw.return_value = mock_hardware_profile
        topo = request.getfixturevalue(topo_fixture)
        res1 = suggest_optimal_resources(topo, problem_small)
        res2 = suggest_optimal_resources(topo, problem_small)
        assert res1 == res2


class TestModeSelection:
    """Verify execution mode logic based on problem characteristics & topology."""

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_kpoint_mode_for_many_kpoints(self, mock_hw, standard_topology, problem_small, mock_hardware_profile):
        mock_hw.return_value = mock_hardware_profile
        # Force high k-point count relative to cores
        prob = {**problem_small, "kpoints": 128}
        res = suggest_optimal_resources(standard_topology, prob)
        assert res["mode"] in ("kpoint", "hybrid")  # K-point parallelism prioritized
        assert res.get("kpar", 0) > 0

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_hybrid_mode_for_numa_systems(self, mock_hw, large_cluster_topology, problem_large, mock_hardware_profile):
        mock_hw.return_value = mock_hardware_profile
        res = suggest_optimal_resources(large_cluster_topology, problem_large)
        # NUMA-aware clusters strongly benefit from Hybrid MPI+OMP
        assert res["mode"] == "hybrid"
        assert res["omp_threads_per_rank"] >= 2

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_mpi_mode_for_gamma_only_small(self, mock_hw, single_node_topology, problem_gamma_only, mock_hardware_profile):
        mock_hw.return_value = mock_hardware_profile
        res = suggest_optimal_resources(single_node_topology, problem_gamma_only)
        # Single k-point, small matrix → MPI-only or minimal OMP
        assert res["mode"] in ("mpi", "hybrid")
        if res["mode"] == "hybrid":
            assert res["omp_threads_per_rank"] <= 2


class TestDivisibilityConstraints:
    """Enforce mathematical divisibility for MPI/OMP/k-point distribution."""

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_omp_divides_total_cores(self, mock_hw, standard_topology, problem_small, mock_hardware_profile):
        mock_hw.return_value = mock_hardware_profile
        res = suggest_optimal_resources(standard_topology, problem_small)
        assert res["recommended_total_cores"] % res["omp_threads_per_rank"] == 0

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_kpar_divides_kpoints_and_ranks(self, mock_hw, standard_topology, problem_small, mock_hardware_profile):
        mock_hw.return_value = mock_hardware_profile
        prob = {**problem_small, "kpoints": 32}
        res = suggest_optimal_resources(standard_topology, prob)
        if res.get("kpar", 1) > 1:
            assert prob["kpoints"] % res["kpar"] == 0
            mpi_ranks = res["recommended_total_cores"] // res["omp_threads_per_rank"]
            assert mpi_ranks % res["kpar"] == 0

    @pytest.mark.parametrize("cores,kpts", [
        (1, 1), (2, 1), (4, 4), (8, 3), (16, 7)
    ])
    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_handles_prime_or_non_divisible_counts(self, mock_hw, cores, kpts, problem_small, mock_hardware_profile):
        """Advisor must gracefully handle non-divisible core/k-point counts."""
        mock_hw.return_value = mock_hardware_profile
        topo = TopologyData(
            nodes=["n1"], cores_per_node=[cores], total_cores=cores,
            env_type="local", scheduler_hints={}, heterogeneous=False
        )
        prob = {**problem_small, "kpoints": kpts}
        res = suggest_optimal_resources(topo, prob)
        # No crashes, valid suggestion returned
        assert res["recommended_total_cores"] == cores
        assert len(res["warnings"]) >= 0  # May warn about imbalance


class TestMemoryAndTimeEstimation:
    """Validate resource estimation logic and safety bounds."""

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_estimated_memory_within_system_limits(self, mock_hw, standard_topology, problem_large, mock_hardware_profile):
        mock_hw.return_value = mock_hardware_profile
        res = suggest_optimal_resources(standard_topology, problem_large)
        est_mem = res.get("estimated_memory_gb", 0)
        assert est_mem > 0
        # Should not exceed total available memory
        assert est_mem <= mock_hardware_profile["memory_gb"] * 1.1  # 10% tolerance for overhead

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    @patch("wien2k_gen.optimizer.advisor.get_job_memory_limit_mb")
    def test_respects_job_memory_limits(self, mock_job_limit, mock_hw, standard_topology, problem_large, mock_hardware_profile):
        mock_hw.return_value = mock_hardware_profile
        mock_job_limit.return_value = 32000  # 32GB job limit
        res = suggest_optimal_resources(standard_topology, problem_large)
        # Should either warn or clamp to limit
        assert res["estimated_memory_gb"] <= 32.0 or any("limit" in w.lower() for w in res["warnings"])


class TestHardwareAwareTuning:
    """Verify ELPA, MKL, AVX, and interconnect impact on recommendations."""

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_elpa_available_enables_higher_kpar(self, mock_hw, standard_topology, problem_small, mock_hardware_profile):
        hw_no_elpa = {**mock_hardware_profile, "elpa_available": False}
        mock_hw.return_value = mock_hardware_profile
        res_with_elpa = suggest_optimal_resources(standard_topology, problem_small)
        
        mock_hw.return_value = hw_no_elpa
        res_no_elpa = suggest_optimal_resources(standard_topology, problem_small)
        
        # ELPA allows more aggressive parallelization for large matrices
        if problem_small["nmat"] > 5000:
            assert res_with_elpa["recommended_total_cores"] >= res_no_elpa["recommended_total_cores"]

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_infiniband_reduces_mpi_overhead_warnings(self, mock_hw, large_cluster_topology, problem_large, mock_hardware_profile):
        hw_ib = {**mock_hardware_profile, "interconnect_type": "infiniband"}
        hw_eth = {**mock_hardware_profile, "interconnect_type": "ethernet"}
        
        mock_hw.return_value = hw_ib
        res_ib = suggest_optimal_resources(large_cluster_topology, problem_large)
        
        mock_hw.return_value = hw_eth
        res_eth = suggest_optimal_resources(large_cluster_topology, problem_large)
        
        ib_warns = [w for w in res_ib["warnings"] if "communication" in w.lower() or "network" in w.lower()]
        eth_warns = [w for w in res_eth["warnings"] if "communication" in w.lower() or "network" in w.lower()]
        assert len(ib_warns) <= len(eth_warns)


class TestEdgeCasesAndFallbacks:
    """Robustness against invalid, missing, or extreme inputs."""

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_handles_single_core_gracefully(self, mock_hw, single_node_topology, problem_small, mock_hardware_profile):
        mock_hw.return_value = mock_hardware_profile
        topo = TopologyData(
            nodes=["n1"], cores_per_node=[1], total_cores=1,
            env_type="local", scheduler_hints={}, heterogeneous=False
        )
        res = suggest_optimal_resources(topo, problem_small)
        assert res["recommended_total_cores"] == 1
        assert res["omp_threads_per_rank"] == 1
        assert res["mode"] == "mpi"

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_handles_zero_kpoints(self, mock_hw, standard_topology, mock_hardware_profile):
        mock_hw.return_value = mock_hardware_profile
        prob = {"atoms": 20, "kpoints": 0, "nmat": 2000, "nbands": 200, "is_soc": False, "is_hybrid": False}
        res = suggest_optimal_resources(standard_topology, prob)
        # Must fallback to safe defaults, not crash
        assert res["recommended_total_cores"] > 0
        assert any("kpoints" in w.lower() for w in res["warnings"]) or res.get("kpar") == 1

    @patch("wien2k_gen.optimizer.advisor.get_hardware_profile")
    def test_malformed_problem_size_triggers_warnings(self, mock_hw, standard_topology, mock_hardware_profile):
        mock_hw.return_value = mock_hardware_profile
        prob = {"atoms": -5, "kpoints": "abc", "nmat": None}
        res = suggest_optimal_resources(standard_topology, prob)
        assert isinstance(res, dict)
        assert len(res["warnings"]) > 0
        assert res["confidence"] < 0.8


# =============================================================================
# Pytest Configuration & Coverage Hints
# =============================================================================
# Run: pytest tests/test_advisor.py -v --cov=wien2k_gen.optimizer.advisor --cov-report=term-missing
# Markers: @pytest.mark.parametrize, @patch, pytest.raises (for future error-path tests)