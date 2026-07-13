"""
Production-Grade Tests for core.topology Module.
Covers Topology dataclass, GPUInfo parsing, NUMA node detection,
heterogeneous node detection, and GPU topology detection.
Uses mock subprocess calls to avoid hardware dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest

from forge.core.topology import (
    GPUInfo,
    NodeSpec,
    NUMANode,
    Topology,
    TopologyType,
    TopologyValidationError,
    _detect_nvidia_gpus_topology,
    detect_gpu_topology,
)


@pytest.fixture
def mock_nvidia_smi_output():
    return (
        "Tesla V100-SXM2-32GB, 32510 MiB, 7.0, GPU-1234-abcd, 00000000:00:1E.0, 0\n"
        "Tesla V100-SXM2-32GB, 32510 MiB, 7.0, GPU-5678-efgh, 00000000:00:1F.0, 1\n"
    )


@pytest.fixture
def mock_rocm_smi_output():
    return (
        "GPU[0], AMD Instinct MI250X, VRAM 65520 MB\n"
        "GPU[1], AMD Instinct MI250X, VRAM 65520 MB\n"
    )


@pytest.fixture
def sample_numa_node():
    return NUMANode(
        node_id=0,
        socket_id=0,
        core_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        memory_mb=65536,
        distance_to={0: 10, 1: 21},
        l3_cache_mb=32.0,
    )


@pytest.fixture
def sample_gpu_info():
    return GPUInfo(
        name="Tesla V100-SXM2-32GB",
        memory_mb=32510,
        compute_capability="7.0",
        uuid="GPU-1234-abcd",
        pci_bus="00000000:00:1E.0",
        numa_affinity=0,
    )


class TestTopologyDataclass:
    """Tests for Topology dataclass creation and validation."""

    def test_basic_creation(self):
        topo = Topology(
            nodes=["node01", "node02"],
            cores_per_node=[32, 32],
            env_type="slurm",
        )
        assert topo.total_cores == 64
        assert len(topo.nodes) == 2
        assert topo.env_type == "slurm"

    def test_heterogeneous_detection(self):
        topo = Topology(
            nodes=["n1", "n2", "n3"],
            cores_per_node=[16, 16, 8],
        )
        assert topo.heterogeneous is True

    def test_homogeneous_detection(self):
        topo = Topology(
            nodes=["n1", "n2"],
            cores_per_node=[32, 32],
        )
        assert topo.heterogeneous is False

    def test_single_node_not_heterogeneous(self):
        topo = Topology(
            nodes=["n1"],
            cores_per_node=[16],
        )
        assert topo.heterogeneous is False

    def test_auto_fill_cores_per_node(self):
        topo = Topology(
            nodes=["n1", "n2", "n3"],
            cores_per_node=[1, 1, 1],
        )
        assert topo.total_cores == 3

    def test_validation_mismatched_lengths(self):
        with pytest.raises(TopologyValidationError):
            Topology(
                nodes=["n1", "n2", "n3"],
                cores_per_node=[32, 32],
            )

    def test_validation_negative_cores(self):
        with pytest.raises(TopologyValidationError):
            Topology(
                nodes=["n1"],
                cores_per_node=[-1],
            )

    def test_validation_duplicate_nodes(self):
        with pytest.raises(TopologyValidationError):
            Topology(
                nodes=["n1", "n1"],
                cores_per_node=[16, 16],
            )

    def test_is_homogeneous_mixed_cores(self):
        topo = Topology(
            nodes=["n1", "n2"],
            cores_per_node=[16, 32],
        )
        assert topo.is_homogeneous() is False

    def test_is_homogeneous_uniform(self):
        topo = Topology(
            nodes=["n1", "n2", "n3"],
            cores_per_node=[32, 32, 32],
            memory_per_node=[128000, 128000, 128000],
        )
        assert topo.is_homogeneous() is True

    def test_get_cores_for_node(self):
        topo = Topology(
            nodes=["a", "b", "c"],
            cores_per_node=[8, 16, 32],
        )
        assert topo.get_cores_for_node("b") == 16

    def test_get_cores_for_node_not_found(self):
        topo = Topology(nodes=["a"], cores_per_node=[8])
        with pytest.raises(KeyError):
            topo.get_cores_for_node("z")

    def test_get_total_memory_mb(self):
        topo = Topology(
            nodes=["n1", "n2"],
            cores_per_node=[16, 16],
            memory_per_node=[64000, 64000],
        )
        assert topo.get_total_memory_mb() == 128000

    def test_update_total(self):
        topo = Topology(nodes=["n1", "n2"], cores_per_node=[16, 16])
        assert topo.total_cores == 32
        topo.cores_per_node = [20, 20]
        topo.update_total()
        assert topo.total_cores == 40

    def test_get_mpi_launcher_command(self):
        topo = Topology(
            nodes=["n1", "n2"],
            cores_per_node=[32, 32],
            scheduler_hints={"mpi_launcher": "srun", "numa_aware": True},
        )
        cmd = topo.get_mpi_launcher_command(64, threads_per_rank=2)
        assert "srun" in cmd
        assert "-n" in cmd
        assert "64" in cmd

    def test_optimal_mpi_distribution_cyclic(self):
        topo = Topology(nodes=["n1", "n2"], cores_per_node=[8, 8])
        dist = topo.get_optimal_mpi_distribution(4, threads_per_rank=1, parallelization_mode="cyclic")
        total_assigned = sum(len(v) for v in dist.values())
        assert total_assigned == 4

    def test_optimal_mpi_distribution_block(self):
        topo = Topology(nodes=["n1", "n2"], cores_per_node=[8, 8])
        dist = topo.get_optimal_mpi_distribution(4, threads_per_rank=1, parallelization_mode="block")
        total_assigned = sum(len(v) for v in dist.values())
        assert total_assigned == 4


class TestGPUInfo:
    """Tests for GPUInfo dataclass and serialization."""

    def test_gpu_info_creation(self):
        gpu = GPUInfo(
            name="A100",
            memory_mb=40960,
            compute_capability="8.0",
            uuid="gpu-uuid-1",
            pci_bus="0000:00:1E.0",
            numa_affinity=0,
        )
        assert gpu.name == "A100"
        assert gpu.memory_mb == 40960
        assert gpu.numa_affinity == 0

    def test_gpu_info_to_dict(self, sample_gpu_info):
        d = sample_gpu_info.to_dict()
        assert d["name"] == "Tesla V100-SXM2-32GB"
        assert d["memory_mb"] == 32510

    def test_gpu_info_from_dict(self):
        data = {
            "name": "A100",
            "memory_mb": 40960,
            "compute_capability": "8.0",
            "uuid": "u",
            "pci_bus": "b",
            "numa_affinity": 1,
        }
        gpu = GPUInfo.from_dict(data)
        assert gpu.name == "A100"
        assert gpu.numa_affinity == 1

    @patch("subprocess.run")
    def test_detect_nvidia_gpus(self, mock_run, mock_nvidia_smi_output):
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_nvidia_smi_output)
        gpus = _detect_nvidia_gpus_topology()
        assert len(gpus) == 2
        assert gpus[0].name == "Tesla V100-SXM2-32GB"
        assert gpus[0].memory_mb == 32510
        assert gpus[1].compute_capability == "7.0"

    @patch("subprocess.run")
    def test_detect_nvidia_gpus_failure(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        gpus = _detect_nvidia_gpus_topology()
        assert gpus == []

    @patch("subprocess.run")
    def test_detect_nvidia_gpus_nonzero_returncode(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        gpus = _detect_nvidia_gpus_topology()
        assert gpus == []


class TestNUMANode:
    """Tests for NUMANode dataclass."""

    def test_numa_node_creation(self, sample_numa_node):
        assert sample_numa_node.node_id == 0
        assert len(sample_numa_node.core_ids) == 8
        assert sample_numa_node.memory_mb == 65536

    def test_numa_node_to_dict(self, sample_numa_node):
        d = sample_numa_node.to_dict()
        assert d["node_id"] == 0
        assert d["socket_id"] == 0

    def test_numa_node_from_dict(self):
        data = {"node_id": 1, "socket_id": 1, "core_ids": [8, 9], "memory_mb": 32768}
        n = NUMANode.from_dict(data)
        assert n.node_id == 1
        assert n.core_ids == [8, 9]

    def test_numactl_binding_empty_cores(self):
        node = NUMANode(node_id=0, core_ids=[])
        assert node.get_numactl_binding() == ""

    def test_numactl_binding_with_cores(self, sample_numa_node):
        binding = sample_numa_node.get_numactl_binding()
        assert "--cpunodebind=0" in binding
        assert "--membind=0" in binding


class TestNodeSpec:
    """Tests for NodeSpec dataclass."""

    def test_node_spec_basic(self):
        spec = NodeSpec(hostname="node01", physical_cores=32, logical_cores=64, sockets=2)
        assert spec.hostname == "node01"
        assert spec.cores_per_socket == 16
        assert spec.logical_cores == 64

    def test_node_spec_auto_logical_fix(self):
        spec = NodeSpec(hostname="node01", physical_cores=32, logical_cores=16, sockets=2)
        assert spec.logical_cores == 32

    def test_node_spec_memory_per_core(self, sample_numa_node):
        spec = NodeSpec(
            hostname="node01",
            physical_cores=32,
            logical_cores=64,
            sockets=2,
            memory_total_mb=128000,
            numa_nodes=[sample_numa_node],
        )
        mem_per_core = spec.get_memory_per_core_gb()
        assert mem_per_core > 0

    def test_node_spec_roundtrip(self, sample_numa_node, sample_gpu_info):
        spec = NodeSpec(
            hostname="node01",
            physical_cores=32,
            logical_cores=64,
            sockets=2,
            numa_nodes=[sample_numa_node],
            gpu_info=[sample_gpu_info],
            memory_total_mb=128000,
        )
        d = spec.to_dict()
        spec2 = NodeSpec.from_dict(d)
        assert spec2.hostname == "node01"
        assert len(spec2.numa_nodes) == 1
        assert len(spec2.gpu_info) == 1


class TestGPUTopology:
    """Tests for GPUTopology and GPU detection."""

    @patch("forge.core.topology._detect_nvidia_gpus_topology")
    def test_detect_gpu_topology_nvidia(self, mock_nvidia, sample_gpu_info):
        mock_nvidia.return_value = [sample_gpu_info]
        topo = detect_gpu_topology()
        assert topo.gpu_per_node == 1
        assert topo.multi_gpu is False

    @patch("forge.core.topology._detect_nvidia_gpus_topology", return_value=[])
    @patch("forge.core.topology._detect_amd_gpus_topology", return_value=[])
    @patch("forge.core.topology._detect_sysfs_gpus_topology", return_value=[])
    def test_detect_no_gpus(self, mock_nv, mock_amd, mock_sys):
        topo = detect_gpu_topology()
        assert topo.gpu_per_node == 0
        assert topo.multi_gpu is False


class TestTopologySerialization:
    """Tests for Topology JSON serialization."""

    def test_to_dict_and_from_dict(self):
        topo = Topology(
            nodes=["n1", "n2"],
            cores_per_node=[32, 32],
            env_type="slurm",
            scheduler_hints={"mpi_launcher": "srun"},
        )
        d = topo.to_dict()
        topo2 = Topology.from_dict(d)
        assert topo2.nodes == topo.nodes
        assert topo2.total_cores == topo.total_cores

    def test_roundtrip_with_node_specs(self, sample_numa_node, sample_gpu_info):
        spec = NodeSpec(
            hostname="node01",
            physical_cores=32,
            logical_cores=64,
            sockets=2,
            numa_nodes=[sample_numa_node],
            gpu_info=[sample_gpu_info],
            memory_total_mb=128000,
        )
        topo = Topology(
            nodes=["node01"],
            cores_per_node=[32],
            env_type="local",
            node_specs={"node01": spec},
        )
        d = topo.to_dict()
        topo2 = Topology.from_dict(d)
        assert "node01" in topo2.node_specs
        assert topo2.node_specs["node01"].physical_cores == 32


class TestTopologyTypeDetection:
    """Tests for network topology type detection."""

    def test_default_unknown(self):
        topo = Topology(nodes=["n1"], cores_per_node=[8])
        assert topo.detect_topology_type() == TopologyType.UNKNOWN

    def test_infiniband_fat_tree(self):
        topo = Topology(
            nodes=["n1", "n2"],
            cores_per_node=[16, 16],
            network_topology={"type": "infiniband"},
        )
        assert topo.detect_topology_type() == TopologyType.FAT_TREE

    def test_omnipath_dragonfly(self):
        topo = Topology(
            nodes=["n1", "n2"],
            cores_per_node=[16, 16],
            network_topology={"type": "omnipath"},
        )
        assert topo.detect_topology_type() == TopologyType.DRAGONFLY

    def test_ethernet_star(self):
        topo = Topology(
            nodes=["n1", "n2"],
            cores_per_node=[16, 16],
            network_topology={"type": "ethernet"},
        )
        assert topo.detect_topology_type() == TopologyType.STAR
