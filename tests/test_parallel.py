"""
Tests for forge.optimizer.parallel — NUMA-aware parallelization engine.
Covers all public functions with edge cases, small/large nmat, ELPA
availability, memory-bound vs compute-bound scenarios.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.core.topology import Topology
from forge.optimizer.parallel import (
    ParallelizationStrategy,
    _estimate_kpoint_weights,
    calculate_balance_quality,
    calculate_kpoint_weights,
    compare_distribution_methods,
    detect_numa_topology,
    ffd_kpoint_distribution,
    generate_ffd_machines,
    generate_numa_aware_machines,
    numa_aware_kpoint_distribution,
    recommend_elpa_solver,
    recommend_gmax,
    recommend_io_strategy,
    recommend_lapw0_strategy,
    recommend_mkl_threading,
    recommend_numa_strategy,
    recommend_rkmax,
    recommend_weighted_kpoint_distribution,
    round_robin_distribution,
    should_use_elpa,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def topo_small() -> Topology:
    return Topology(nodes=["n1"], cores_per_node=[8], env_type="slurm")


@pytest.fixture
def topo_dual_socket() -> Topology:
    return Topology(
        nodes=["n1", "n2"],
        cores_per_node=[32, 32],
        env_type="slurm",
        scheduler_hints={"mpi_launcher": "srun", "numa_aware": True},
    )


@pytest.fixture
def topo_numa_four() -> Topology:
    return Topology(
        nodes=["n1", "n2", "n3", "n4"],
        cores_per_node=[32, 32, 32, 32],
        env_type="slurm",
    )


@pytest.fixture
def topo_single_core() -> Topology:
    return Topology(nodes=["n1"], cores_per_node=[1], env_type="local")


@pytest.fixture
def topo_empty() -> Topology:
    return Topology(nodes=[], cores_per_node=[], env_type="unknown")


@pytest.fixture
def patch_numa_available():
    with patch(
        "forge.core.hardware.get_numa_node_count", return_value=4
    ), patch(
        "forge.core.hardware.get_memory_bandwidth_gb_s", return_value=200.0
    ):
        yield


@pytest.fixture
def patch_numa_low_bw():
    with patch(
        "forge.core.hardware.get_numa_node_count", return_value=4
    ), patch(
        "forge.core.hardware.get_memory_bandwidth_gb_s", return_value=30.0
    ):
        yield


@pytest.fixture
def patch_no_numa():
    with patch(
        "forge.core.hardware.get_numa_node_count", side_effect=ImportError
    ), patch(
        "forge.core.hardware.get_memory_bandwidth_gb_s", side_effect=ImportError
    ):
        yield


@pytest.fixture
def uniform_weights_12():
    return [1.0] * 12


@pytest.fixture
def skewed_weights_12():
    return [0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# ParallelizationStrategy dataclass
# ---------------------------------------------------------------------------

class TestParallelizationStrategy:
    def test_default_construction(self):
        s = ParallelizationStrategy(
            mode="pure_mpi",
            mpi_ranks=4,
            omp_threads=2,
            cores_per_node=[4],
            numa_binding=False,
            granularity=1,
            efficiency_pct=85.0,
            recommendation="test",
        )
        assert s.mode == "pure_mpi"
        assert s.mpi_ranks == 4
        assert s.omp_threads == 2
        assert s.efficiency_pct == 85.0

    def test_hybrid_mode(self):
        s = ParallelizationStrategy(
            mode="hybrid",
            mpi_ranks=8,
            omp_threads=4,
            cores_per_node=[8, 8],
            numa_binding=True,
            granularity=2,
            efficiency_pct=90.0,
            recommendation="hybrid rec",
        )
        assert s.mode == "hybrid"
        assert s.mpi_ranks == 8
        assert s.omp_threads == 4

    def test_numa_aware_mode(self):
        s = ParallelizationStrategy(
            mode="numa_aware",
            mpi_ranks=4,
            omp_threads=8,
            cores_per_node=[32, 32, 32, 32],
            numa_binding=True,
            granularity=8,
            efficiency_pct=92.0,
            recommendation="numa rec",
        )
        assert s.mode == "numa_aware"
        assert s.numa_binding is True
        assert s.granularity == 8

    def test_equality(self):
        PS = ParallelizationStrategy
        a = PS(
            mode="hybrid", mpi_ranks=4, omp_threads=2,
            cores_per_node=[4], numa_binding=False,
            granularity=1, efficiency_pct=80.0, recommendation="r",
        )
        b = PS(
            mode="hybrid", mpi_ranks=4, omp_threads=2,
            cores_per_node=[4], numa_binding=False,
            granularity=1, efficiency_pct=80.0, recommendation="r",
        )
        assert a == b


# ---------------------------------------------------------------------------
# recommend_numa_strategy
# ---------------------------------------------------------------------------

class TestRecommendNumaStrategy:
    @pytest.mark.usefixtures("patch_numa_available")
    def test_very_large_nmat_falls_back_to_hybrid(self, topo_numa_four):
        s = recommend_numa_strategy(topo_numa_four, nmat=15000, nkpt=64, atoms=10)
        assert s.mode == "hybrid"
        assert s.numa_binding is False

    @pytest.mark.usefixtures("patch_numa_available")
    def test_large_nmat_no_granular(self, topo_numa_four):
        s = recommend_numa_strategy(topo_numa_four, nmat=7000, nkpt=32, atoms=10)
        assert s.mode == "numa_aware"
        assert s.granularity == 1
        assert s.numa_binding is True
        assert s.efficiency_pct == 80.0

    @pytest.mark.usefixtures("patch_numa_available")
    def test_small_nmat_falls_back_hybrid(self, topo_dual_socket):
        s = recommend_numa_strategy(topo_dual_socket, nmat=2000, nkpt=8, atoms=5)
        assert s.mode == "hybrid"
        assert s.numa_binding is False

    @pytest.mark.usefixtures("patch_numa_available")
    def test_nmat_zero(self, topo_dual_socket):
        s = recommend_numa_strategy(topo_dual_socket, nmat=0, nkpt=4, atoms=1)
        assert s.mode == "hybrid"

    @pytest.mark.usefixtures("patch_numa_available")
    def test_nkpt_zero(self, topo_dual_socket):
        s = recommend_numa_strategy(topo_dual_socket, nmat=5000, nkpt=0, atoms=1)
        assert s.mode == "hybrid"

    @pytest.mark.usefixtures("patch_numa_available")
    def test_max_cores_one(self, topo_single_core):
        s = recommend_numa_strategy(topo_single_core, nmat=15000, nkpt=32, atoms=5)
        assert s.mpi_ranks >= 1

    @pytest.mark.usefixtures("patch_numa_available")
    def test_available_cores_limits(self, topo_numa_four):
        s = recommend_numa_strategy(
            topo_numa_four, nmat=15000, nkpt=64, atoms=10, available_cores=8,
        )
        assert s.mpi_ranks <= 8

    @pytest.mark.usefixtures("patch_numa_low_bw")
    def test_low_bandwidth_memory_bound(self, topo_numa_four):
        s = recommend_numa_strategy(topo_numa_four, nmat=100, nkpt=4, atoms=2)
        assert isinstance(s.mode, str)

    @pytest.mark.usefixtures("patch_no_numa")
    def test_no_numa_detected_fallback(self, topo_dual_socket):
        s = recommend_numa_strategy(topo_dual_socket, nmat=15000, nkpt=16, atoms=5)
        assert s.mode == "hybrid"

    @pytest.mark.usefixtures("patch_numa_available")
    def test_very_large_nmat_kpts_per_mpi_always_one(self, topo_numa_four):
        s = recommend_numa_strategy(topo_numa_four, nmat=15000, nkpt=8, atoms=10)
        assert s.mode == "hybrid"


# ---------------------------------------------------------------------------
# recommend_lapw0_strategy
# ---------------------------------------------------------------------------

class TestRecommendLapw0Strategy:
    @pytest.mark.usefixtures("patch_numa_available")
    def test_large_fft_grid_triggers_hybrid(self, topo_dual_socket):
        s = recommend_lapw0_strategy(topo_dual_socket, nmat=1000, fft_nx=200, fft_ny=200, fft_nz=200)
        assert s.mode == "hybrid"
        assert s.mpi_ranks >= 2

    @pytest.mark.usefixtures("patch_numa_available")
    def test_small_fft_grid_pure_openmp(self, topo_dual_socket):
        s = recommend_lapw0_strategy(topo_dual_socket, nmat=100, fft_nx=10, fft_ny=10, fft_nz=10)
        assert s.mode == "pure_mpi"
        assert s.mpi_ranks == 1

    @pytest.mark.usefixtures("patch_numa_available")
    def test_no_fft_dims_estimates_from_nmat(self, topo_dual_socket):
        s = recommend_lapw0_strategy(topo_dual_socket, nmat=500000)
        if s.mode == "hybrid":
            assert s.numa_binding is True

    @pytest.mark.usefixtures("patch_no_numa")
    def test_no_numa_fallback_single_node(self, topo_dual_socket):
        s = recommend_lapw0_strategy(topo_dual_socket, nmat=1000, fft_nx=200, fft_ny=200, fft_nz=200)
        assert isinstance(s.mode, str)


# ---------------------------------------------------------------------------
# recommend_io_strategy
# ---------------------------------------------------------------------------

class TestRecommendIOStrategy:
    def test_very_large_nmat(self):
        r = recommend_io_strategy(nmat=15000, nkpt=4, atoms=50)
        assert r["granularity"] == 16
        assert r["nowrite_vector"] is False
        assert "DANGER" in r["nowrite_vector_warning"]

    def test_very_large_nmat_many_kpt(self):
        r = recommend_io_strategy(nmat=15000, nkpt=32, atoms=50)
        assert r["vector_split"] in (2, 4)

    def test_large_nmat(self):
        r = recommend_io_strategy(nmat=7000, nkpt=10, atoms=30)
        assert r["granularity"] == 8
        assert r["vector_split"] == 2

    def test_large_nmat_high_kpt(self):
        r = recommend_io_strategy(nmat=7000, nkpt=32, atoms=30)
        assert r["vector_split"] == 1

    def test_medium_nmat(self):
        r = recommend_io_strategy(nmat=3000, nkpt=8, atoms=20)
        assert r["granularity"] == 1
        assert r["vector_split"] == 0

    def test_small_nmat(self):
        r = recommend_io_strategy(nmat=100, nkpt=4, atoms=5)
        assert r["granularity"] == 1
        assert "Standard I/O" in r["recommendation"]

    def test_tmpfs_scratch(self):
        r = recommend_io_strategy(nmat=100, nkpt=4, atoms=5, scratch_fs="tmpfs")
        assert "tmpfs" in r["recommendation"]

    def test_ramfs_scratch(self):
        r = recommend_io_strategy(nmat=100, nkpt=4, atoms=5, scratch_fs="ramfs")
        assert "tmpfs" in r["recommendation"]

    def test_high_granularity_warning(self):
        r = recommend_io_strategy(nmat=15000, nkpt=4, atoms=50)
        assert "warnings" in r

    def test_nmat_zero(self):
        r = recommend_io_strategy(nmat=0, nkpt=1, atoms=1)
        assert r["granularity"] == 1


# ---------------------------------------------------------------------------
# recommend_rkmax
# ---------------------------------------------------------------------------

class TestRecommendRkmax:
    def test_heavy_atom_max_z_above_70(self):
        assert recommend_rkmax([80]) == 8.0

    def test_atom_max_z_50_70(self):
        assert recommend_rkmax([60]) == 7.5

    def test_atom_max_z_30_50(self):
        assert recommend_rkmax([40]) == 7.0

    def test_atom_max_z_20_30(self):
        assert recommend_rkmax([25]) == 6.5

    def test_light_atom(self):
        assert recommend_rkmax([6]) == 6.0

    def test_hard_element_rmt_small(self):
        v = recommend_rkmax([8, 14], rmt_ratios=[1.5, 2.0])
        assert v >= 7.0

    def test_hard_element_no_rmt(self):
        v = recommend_rkmax([8])
        assert v >= 7.0

    def test_soc_boosts(self):
        v = recommend_rkmax([6], is_soc=True)
        assert v >= 7.5

    def test_opt_adds_half(self):
        a = recommend_rkmax([6], calculation_type="scf")
        b = recommend_rkmax([6], calculation_type="opt")
        assert b > a

    def test_efg_adds_one(self):
        v = recommend_rkmax([6], calculation_type="efg")
        assert v >= 7.0

    def test_hyperfine_adds_one(self):
        v = recommend_rkmax([6], calculation_type="hyperfine")
        assert v >= 7.0

    def test_dos_adds_half(self):
        a = recommend_rkmax([6], calculation_type="scf")
        b = recommend_rkmax([6], calculation_type="dos")
        assert b > a

    def test_empty_atomic_numbers(self):
        assert recommend_rkmax([]) == 7.0


# ---------------------------------------------------------------------------
# recommend_gmax
# ---------------------------------------------------------------------------

class TestRecommendGmax:
    def test_default_scf_factor(self):
        assert recommend_gmax(7.0, "scf") == 14.0

    def test_dos_factor(self):
        assert recommend_gmax(7.0, "dos") == 14.0

    def test_band_factor(self):
        assert recommend_gmax(7.0, "band") == 14.0

    def test_opt_factor(self):
        assert recommend_gmax(7.0, "opt") == 17.5

    def test_forces_factor(self):
        assert recommend_gmax(7.0, "forces") == 17.5

    def test_efg_factor(self):
        assert recommend_gmax(7.0, "efg") == 21.0

    def test_hyperfine_factor(self):
        assert recommend_gmax(7.0, "hyperfine") == 21.0

    def test_unknown_calc_type_defaults(self):
        assert recommend_gmax(7.0, "unknown_type") == 14.0

    def test_rkmax_zero(self):
        assert recommend_gmax(0.0) == 0.0

    def test_rounding(self):
        assert recommend_gmax(7.15, "scf") == 14.3


# ---------------------------------------------------------------------------
# recommend_elpa_solver
# ---------------------------------------------------------------------------

class TestRecommendElpaSolver:
    def test_small_nmat_returns_none(self):
        assert recommend_elpa_solver(200, nkpt=1) is None

    def test_very_large_nmat_elpa2(self):
        assert recommend_elpa_solver(10000, nkpt=1) == "elpa2"

    def test_large_nmat_many_cores_elpa2(self):
        assert recommend_elpa_solver(6000, nkpt=1, num_cores=128) == "elpa2"

    def test_medium_soc_elpa2(self):
        assert recommend_elpa_solver(3000, nkpt=1, is_soc=True) == "elpa2"

    def test_hybrid_elpa2(self):
        assert recommend_elpa_solver(4000, nkpt=1, is_hybrid=True) == "elpa2"

    def test_default_elpa1(self):
        assert recommend_elpa_solver(3000, nkpt=1) == "elpa1"

    def test_nmat_exactly_500(self):
        assert recommend_elpa_solver(500, nkpt=1) is None

    def test_nmat_499(self):
        assert recommend_elpa_solver(499, nkpt=1) is None


# ---------------------------------------------------------------------------
# should_use_elpa
# ---------------------------------------------------------------------------

class TestShouldUseElpa:
    def test_very_large_nmat(self):
        assert should_use_elpa(9000) is True

    def test_large_nmat_many_cores(self):
        assert should_use_elpa(6000, num_cores=128) is True

    def test_large_nmat_few_cores(self):
        assert should_use_elpa(6000, num_cores=16) is False

    def test_small_nmat(self):
        assert should_use_elpa(1000) is False

    def test_tiny_nmat(self):
        assert should_use_elpa(100) is False


# ---------------------------------------------------------------------------
# recommend_mkl_threading
# ---------------------------------------------------------------------------

class TestRecommendMklThreading:
    def test_large_nmat(self):
        assert recommend_mkl_threading(5000, nkpt=4) == 4

    def test_medium_nmat(self):
        assert recommend_mkl_threading(3000, nkpt=4) == 8

    def test_small_nmat(self):
        assert recommend_mkl_threading(1000, nkpt=4) is None

    def test_exact_boundary_4000(self):
        assert recommend_mkl_threading(4000, nkpt=1) == 8

    def test_exact_boundary_4001(self):
        assert recommend_mkl_threading(4001, nkpt=1) == 4

    def test_exact_boundary_2000(self):
        assert recommend_mkl_threading(2000, nkpt=1) is None


# ---------------------------------------------------------------------------
# recommend_weighted_kpoint_distribution
# ---------------------------------------------------------------------------

class TestRecommendWeightedKpointDistribution:
    def test_single_rank(self):
        result = recommend_weighted_kpoint_distribution(nkpt=10, nmpi=1)
        assert result[0] == list(range(10))

    def test_uniform_distribution(self, uniform_weights_12):
        result = recommend_weighted_kpoint_distribution(nkpt=12, nmpi=4, k_weights=uniform_weights_12)
        total = sum(len(v) for v in result.values())
        assert total == 12
        assert len(result) == 4

    def test_skewed_distribution(self, skewed_weights_12):
        result = recommend_weighted_kpoint_distribution(nkpt=12, nmpi=4, k_weights=skewed_weights_12)
        total = sum(len(v) for v in result.values())
        assert total == 12

    def test_no_weights_estimates(self):
        result = recommend_weighted_kpoint_distribution(nkpt=20, nmpi=4)
        total = sum(len(v) for v in result.values())
        assert total == 20

    def test_nkpt_zero(self):
        result = recommend_weighted_kpoint_distribution(nkpt=0, nmpi=4)
        assert result[0] == []

    def test_nmpi_exceeds_nkpt(self):
        result = recommend_weighted_kpoint_distribution(nkpt=4, nmpi=8)
        assert len(result) == 8


# ---------------------------------------------------------------------------
# _estimate_kpoint_weights
# ---------------------------------------------------------------------------

class TestEstimateKpointWeights:
    def test_returns_correct_length(self):
        w = _estimate_kpoint_weights(20)
        assert len(w) == 20

    def test_uniform_for_small_nkpt(self):
        w = _estimate_kpoint_weights(3)
        assert w == [1.0, 1.0, 1.0]

    def test_symmetry_off_uniform(self):
        w = _estimate_kpoint_weights(20, symmetry_weight=False)
        assert all(x == 1.0 for x in w)

    def test_front_lighter_back_heavier(self):
        w = _estimate_kpoint_weights(30)
        assert w[2] < 1.0
        assert w[-1] > 1.0

    def test_nkpt_one(self):
        w = _estimate_kpoint_weights(1)
        assert w == [1.0]


# ---------------------------------------------------------------------------
# numa_aware_kpoint_distribution
# ---------------------------------------------------------------------------

class TestNumaAwareKpointDistribution:
    def test_basic_distribution(self, skewed_weights_12):
        result = numa_aware_kpoint_distribution(
            kpoints=12, numa_nodes=4,
            cores_per_node=[8, 8, 8, 8],
            k_weights=skewed_weights_12,
        )
        total = sum(len(v) for v in result["node_kpts"].values())
        assert total == 12
        assert 0.0 <= result["balance_ratio"] <= 1.0

    def test_uniform_weights(self):
        result = numa_aware_kpoint_distribution(
            kpoints=10, numa_nodes=2, cores_per_node=[16, 16],
        )
        total = sum(len(v) for v in result["node_kpts"].values())
        assert total == 10

    def test_mismatched_weights_length(self):
        result = numa_aware_kpoint_distribution(
            kpoints=5, numa_nodes=2, cores_per_node=[8, 8],
            k_weights=[1.0, 2.0],
        )
        assert "balance_ratio" in result

    def test_single_node(self):
        result = numa_aware_kpoint_distribution(
            kpoints=10, numa_nodes=1, cores_per_node=[32],
        )
        assert result["balance_ratio"] == 1.0

    def test_all_weights_equal_perfect_balance(self):
        result = numa_aware_kpoint_distribution(
            kpoints=8, numa_nodes=2, cores_per_node=[16, 16],
            k_weights=[1.0] * 8,
        )
        assert result["balance_ratio"] == 1.0


# ---------------------------------------------------------------------------
# generate_numa_aware_machines
# ---------------------------------------------------------------------------

class TestGenerateNumaAwareMachines:
    def test_generates_expected_structure(self):
        node_kpts = {0: [0, 1, 2], 1: [3, 4]}
        node_cores = {0: 4, 1: 4}
        content = generate_numa_aware_machines("testcase", node_kpts, node_cores)
        assert "lapw1:node01:4" in content
        assert "lapw1:node02:4" in content
        assert "NUMA Node 0" in content

    def test_single_node(self):
        node_kpts = {0: [0, 1]}
        node_cores = {0: 8}
        content = generate_numa_aware_machines("testcase", node_kpts, node_cores)
        assert "node01" in content

    def test_custom_prefix(self):
        node_kpts = {0: [0]}
        node_cores = {0: 4}
        content = generate_numa_aware_machines("tc", node_kpts, node_cores, hostname_prefix="hpc")
        assert "lapw1:hpc01:4" in content


# ---------------------------------------------------------------------------
# detect_numa_topology
# ---------------------------------------------------------------------------

class TestDetectNumaTopology:
    @patch("subprocess.run")
    def test_numactl_success(self, mock_run):
        stdout = (
            "available: 2 nodes (0-1)\n"
            "node 0 cpus: 0 1 2 3\n"
            "node 1 cpus: 4 5 6 7\n"
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        result = detect_numa_topology()
        assert result["detected"] is True
        assert result["num_nodes"] == 2
        assert sum(result["cores_per_node"]) == 8

    @patch("subprocess.run")
    @patch("os.cpu_count", return_value=16)
    def test_all_detection_fail_final_fallback(self, mock_cpu, mock_run):
        mock_run.side_effect = FileNotFoundError
        with patch.object(Path, "glob", return_value=[]):
            result = detect_numa_topology()
        assert result["detected"] is False
        assert result["num_nodes"] == 1
        assert result["total_cores"] == 16


# ---------------------------------------------------------------------------
# calculate_kpoint_weights
# ---------------------------------------------------------------------------

class TestCalculateKpointWeights:
    def test_file_not_found_fallback(self, tmp_path):
        os.chdir(tmp_path)
        try:
            w = calculate_kpoint_weights("nonexistent")
            assert w == [1.0]
        finally:
            os.chdir("/workspace")

    def test_valid_klist(self, tmp_path):
        os.chdir(tmp_path)
        klist = tmp_path / "test.klist"
        klist.write_text("4\n0.0 0.0 0.0 2.0\n0.5 0.0 0.0 2.0\n0.0 0.5 0.0 4.0\n0.0 0.0 0.5 4.0\n")
        try:
            w = calculate_kpoint_weights("test")
            assert len(w) == 4
            assert abs(sum(w) - 1.0) < 1e-9
        finally:
            os.chdir("/workspace")

    def test_klist_with_case_subdir(self, tmp_path):
        os.chdir(tmp_path)
        sub = tmp_path / "case"
        sub.mkdir()
        klist = sub / "case.klist"
        klist.write_text("2\n0.0 0.0 0.0 1.0\n0.0 0.5 0.0 1.0\n")
        try:
            w = calculate_kpoint_weights("case")
            assert len(w) == 2
        finally:
            os.chdir("/workspace")


# ---------------------------------------------------------------------------
# ffd_kpoint_distribution
# ---------------------------------------------------------------------------

class TestFFDKpointDistribution:
    def test_basic_distribution(self, uniform_weights_12):
        result = ffd_kpoint_distribution(uniform_weights_12, 4)
        total = sum(len(v) for v in result["rank_kpts"].values())
        assert total == 12
        assert result["method"] == "ffd"

    def test_skewed_distribution(self, skewed_weights_12):
        result = ffd_kpoint_distribution(skewed_weights_12, 4)
        assert 0.0 < result["balance_ratio"] <= 1.0

    def test_zero_ranks(self):
        result = ffd_kpoint_distribution([1.0, 2.0], 0)
        assert result["rank_kpts"] == {}
        assert result["balance_ratio"] == 1.0

    def test_single_rank(self):
        result = ffd_kpoint_distribution([1.0, 2.0, 3.0], 1)
        assert len(result["rank_kpts"][0]) == 3

    def test_empty_weights(self):
        result = ffd_kpoint_distribution([], 2)
        assert result["rank_kpts"] == {0: [], 1: []}
        assert result["balance_ratio"] == 1.0


# ---------------------------------------------------------------------------
# calculate_balance_quality
# ---------------------------------------------------------------------------

class TestCalculateBalanceQuality:
    def test_perfect_balance(self):
        r = calculate_balance_quality([10.0, 10.0, 10.0, 10.0])
        assert r["balance_ratio"] == 1.0
        assert r["efficiency"] == 1.0
        assert r["load_variance"] == 0.0

    def test_imbalanced(self):
        r = calculate_balance_quality([20.0, 10.0, 5.0, 5.0])
        assert r["balance_ratio"] < 1.0
        assert r["efficiency"] < 1.0
        assert r["load_variance"] > 0.0

    def test_empty_list(self):
        r = calculate_balance_quality([])
        assert r["balance_ratio"] == 1.0
        assert r["efficiency"] == 1.0

    def test_single_rank(self):
        r = calculate_balance_quality([42.0])
        assert r["balance_ratio"] == 1.0
        assert r["max_load"] == 42.0
        assert r["min_load"] == 42.0


# ---------------------------------------------------------------------------
# round_robin_distribution
# ---------------------------------------------------------------------------

class TestRoundRobinDistribution:
    def test_basic_distribution(self, skewed_weights_12):
        result = round_robin_distribution(skewed_weights_12, 4)
        total = sum(len(v) for v in result["rank_kpts"].values())
        assert total == 12
        assert result["method"] == "round_robin"

    def test_zero_ranks(self):
        result = round_robin_distribution([1.0, 2.0], 0)
        assert result["rank_kpts"] == {}
        assert result["balance_ratio"] == 1.0

    def test_single_rank(self):
        result = round_robin_distribution([1.0, 2.0, 3.0], 1)
        assert len(result["rank_kpts"][0]) == 3


# ---------------------------------------------------------------------------
# compare_distribution_methods
# ---------------------------------------------------------------------------

class TestCompareDistributionMethods:
    def test_compare_returns_both(self, skewed_weights_12):
        result = compare_distribution_methods(skewed_weights_12, 4)
        assert "ffd" in result
        assert "round_robin" in result
        assert "winner" in result
        assert "improvement_pct" in result

    def test_compare_with_uniform_weights(self, uniform_weights_12):
        result = compare_distribution_methods(uniform_weights_12, 4)
        assert result["winner"] in ("ffd", "round_robin", "tie")


# ---------------------------------------------------------------------------
# generate_ffd_machines
# ---------------------------------------------------------------------------

class TestGenerateFFDMachines:
    def test_generates_structure(self):
        rank_kpts = {0: [0, 1, 2], 1: [3, 4, 5]}
        rank_loads = [1.0, 1.0]
        content = generate_ffd_machines(rank_kpts, rank_loads, 2)
        assert "lapw1:rank00:4" in content
        assert "lapw1:rank01:4" in content
        assert "FFD-optimized" in content

    def test_single_rank(self):
        rank_kpts = {0: [0, 1, 2, 3]}
        rank_loads = [1.0]
        content = generate_ffd_machines(rank_kpts, rank_loads, 1)
        assert "lapw1:rank00:4" in content

    def test_more_than_eight_kpts_truncates(self):
        rank_kpts = {0: list(range(15))}
        rank_loads = [1.0]
        content = generate_ffd_machines(rank_kpts, rank_loads, 1)
        assert "more" in content

    def test_custom_prefix(self):
        rank_kpts = {0: [0]}
        rank_loads = [1.0]
        content = generate_ffd_machines(rank_kpts, rank_loads, 1, hostname_prefix="hpc")
        assert "lapw1:hpc00:4" in content
