"""Tests for GPU recommendation and GPU-aware .machines generation."""

from unittest.mock import patch

from forge.backends.gpu_backend import generate_gpu_machines, get_gpu_recommendation
from forge.core.topology import GPUInfo, GPUTopology, Topology

_PCIE = {
    "nvlink_active": False,
    "infinity_fabric_active": False,
    "interconnect_type": "pcie",
    "bandwidth_estimate_gb_s": 32.0,
}

_NVLINK = {
    "nvlink_active": True,
    "infinity_fabric_active": False,
    "interconnect_type": "nvlink",
    "bandwidth_estimate_gb_s": 600.0,
}


def _gpu(name: str = "NVIDIA A100", memory_mb: int = 40000) -> GPUInfo:
    return GPUInfo(name=name, memory_mb=memory_mb, compute_capability="8.0")


def _topo(nodes, cores, gpus):
    return Topology(
        nodes=list(nodes),
        cores_per_node=list(cores),
        env_type="slurm",
        gpu_topology=GPUTopology(gpus=list(gpus), gpu_per_node=len(gpus), multi_gpu=len(gpus) > 1),
    )


def _gpu_rec(**overrides):
    rec = {
        "use_gpu": True,
        "gpu_count": 4,
        "cuda_visible": "CUDA_VISIBLE_DEVICES=0,1,2,3",
        "gpu_per_mpi_rank": 1,
        "reason": "test",
        "recommended_library": "cuSOLVER",
        "interconnect_info": _PCIE,
    }
    rec.update(overrides)
    return rec


class TestGpuPerRankDenominator:
    @patch("forge.backends.gpu_backend.get_gpu_interconnect_info", return_value=_NVLINK)
    def test_uses_mpi_ranks_not_cpu_cores(self, _ic):
        topo = _topo(["n01"], [100], [_gpu() for _ in range(4)])
        rec = get_gpu_recommendation(topo, nmat=8000, nkpt=8, mode="mpi", mpi_ranks=2)
        assert rec["use_gpu"] is True
        assert rec["gpu_per_mpi_rank"] == 2

    @patch("forge.backends.gpu_backend.get_gpu_interconnect_info", return_value=_NVLINK)
    def test_four_ranks_and_four_gpus_is_one_each(self, _ic):
        topo = _topo(["n01"], [100], [_gpu() for _ in range(4)])
        rec = get_gpu_recommendation(topo, nmat=8000, nkpt=8, mode="mpi", mpi_ranks=4)
        assert rec["gpu_per_mpi_rank"] == 1

    @patch("forge.backends.gpu_backend.get_gpu_interconnect_info", return_value=_NVLINK)
    def test_unknown_rank_count_defaults_to_one(self, _ic):
        topo = _topo(["n01"], [100], [_gpu() for _ in range(4)])
        rec = get_gpu_recommendation(topo, nmat=8000, nkpt=8, mode="mpi")
        assert rec["gpu_per_mpi_rank"] == 1

    @patch("forge.backends.gpu_backend.get_gpu_interconnect_info", return_value=_NVLINK)
    def test_hybrid_mode_always_one_gpu_per_rank(self, _ic):
        topo = _topo(["n01"], [100], [_gpu() for _ in range(8)])
        rec = get_gpu_recommendation(topo, nmat=8000, nkpt=8, mode="hybrid", mpi_ranks=2)
        assert rec["gpu_per_mpi_rank"] == 1

    @patch("forge.backends.gpu_backend.get_gpu_interconnect_info", return_value=_NVLINK)
    def test_does_not_collapse_to_one_for_any_core_gpu_ratio(self, _ic):
        topo = _topo(["n01"], [100], [_gpu() for _ in range(8)])
        rec = get_gpu_recommendation(topo, nmat=8000, nkpt=8, mode="mpi", mpi_ranks=2)
        assert rec["gpu_per_mpi_rank"] == 4

    @patch("forge.backends.gpu_backend.get_gpu_interconnect_info", return_value=_NVLINK)
    def test_more_ranks_than_gpus_shares_round_robin(self, _ic):
        topo = _topo(["n01"], [100], [_gpu() for _ in range(4)])
        rec = get_gpu_recommendation(topo, nmat=8000, nkpt=8, mode="mpi", mpi_ranks=8)
        assert rec["gpu_per_mpi_rank"] == 0


class TestGenerateGpuMachinesAllocation:
    def test_uses_recommended_total_cores_not_raw_topology(self):
        topo = _topo(["n01", "n02", "n03", "n04"], [8, 8, 8, 8], [_gpu() for _ in range(4)])
        content = generate_gpu_machines(
            topo,
            {
                "backend": "wien2k",
                "recommended_total_cores": 2,
                "gpu_recommendation": _gpu_rec(),
            },
        )
        assert "n01: 0" not in content
        assert "n02: 0" not in content
        assert "n03: 0" not in content
        assert "n04: 0" not in content
        described = 0
        for line in content.splitlines():
            if line.startswith("lapw1:"):
                described += int(line.split(":")[2].split()[0])
        assert described == 2
        assert content.count("lapw1:") == 2

    def test_heterogeneous_small_budget_drops_nodes(self):
        topo = _topo(["gpu01", "cpu02"], [16, 8], [_gpu() for _ in range(2)])
        content = generate_gpu_machines(
            topo,
            {
                "backend": "wien2k",
                "recommended_total_cores": 1,
                "gpu_recommendation": _gpu_rec(gpu_count=2),
            },
        )
        for line in content.splitlines():
            if line.startswith("lapw1:"):
                cores = int(line.split(":")[2].split()[0])
                assert cores > 0
        assert "excluded" in content
        assert content.count("lapw1:") == 1

    def test_granularity_and_omp_from_suggestion(self):
        topo = _topo(["n01"], [8], [_gpu()])
        content = generate_gpu_machines(
            topo,
            {
                "backend": "wien2k",
                "recommended_total_cores": 8,
                "granularity": 4,
                "omp_threads_per_rank": 2,
                "gpu_recommendation": _gpu_rec(gpu_count=1),
            },
        )
        assert "granularity: 4" in content
        assert "omp_global: 2" in content

    def test_no_gpu_returns_empty(self):
        topo = _topo(["n01"], [8], [_gpu()])
        assert generate_gpu_machines(topo, {"gpu_recommendation": {"use_gpu": False}}) == ""

    def test_round_robin_when_gpu_per_rank_is_zero(self):
        topo = _topo(["n01", "n02", "n03"], [4, 4, 4], [_gpu() for _ in range(2)])
        content = generate_gpu_machines(
            topo,
            {
                "backend": "wien2k",
                "recommended_total_cores": 12,
                "gpu_recommendation": _gpu_rec(gpu_count=2, gpu_per_mpi_rank=0),
            },
        )
        gpu_lines = [ln for ln in content.splitlines() if "gpu:" in ln and ln.startswith("lapw1:")]
        assert len(gpu_lines) == 2
        cpu_only = [
            ln
            for ln in content.splitlines()
            if ln.startswith("lapw1:") and "gpu:" not in ln
        ]
        assert len(cpu_only) == 1
