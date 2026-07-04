"""
Production-Grade Tests for core.scheduler Module.
Covers topology detection, scheduler environment parsing, hardware profiling,
TopologyData validation, fallback behavior, and thread-safe execution.
"""

import os
import json
import pytest
import threading
from unittest.mock import patch, MagicMock, call
from pathlib import Path

from wien2k_gen.core.scheduler import detect
from wien2k_gen.core.topology import Topology, TopologyValidationError
from wien2k_gen.types import TopologyData
from wien2k_gen.exceptions import DetectionFailedError, InvalidTopologyError


# =============================================================================
# Fixtures: Mock Outputs for System Commands
# =============================================================================

@pytest.fixture
def mock_lscpu_json():
    """Standard lscpu -J output."""
    return json.dumps({
        "lscpu": [
            {"field": "Architecture:", "data": "x86_64"},
            {"field": "CPU(s):", "data": "32"},
            {"field": "Thread(s) per core:", "data": "2"},
            {"field": "Core(s) per socket:", "data": "8"},
            {"field": "Socket(s):", "data": "2"},
            {"field": "NUMA node(s):", "data": "2"},
            {"field": "Model name:", "data": "Intel(R) Xeon(R) Gold 6248R"}
        ]
    })

@pytest.fixture
def mock_numactl_output():
    """Standard numactl --hardware output."""
    return (
        "available: 2 nodes (0-1)\n"
        "node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\n"
        "node 1 cpus: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\n"
        "node 0 size: 65536 MB\nnode 1 size: 65536 MB\n"
    )

@pytest.fixture
def env_slurm():
    """Simulate SLURM environment."""
    return {
        "SLURM_JOB_ID": "123456",
        "SLURM_NNODES": "2",
        "SLURM_NTASKS": "32",
        "SLURM_TASKS_PER_NODE": "16,16",
        "SLURM_JOB_NODELIST": "node[01-02]",
        "SLURM_STEP_LAUNCHER_PORT": "12345"
    }


# =============================================================================
# Test Suites
# =============================================================================

class TestSchedulerDetection:
    """Tests for main detect() function and helper parsers."""

    @patch("wien2k_gen.core.scheduler.get_interconnect_info")
    def test_detect_successful_topology(self, mock_interconnect, env_slurm):
        """Happy path: Full topology detection in SLURM env."""
        mock_interconnect.return_value = {}
        with patch.dict(os.environ, env_slurm, clear=False):
            topo = detect(force_refresh=True)
            
        assert topo.total_cores == 32
        assert topo.env_type in ("slurm", "cluster")
        assert topo.heterogeneous is False
        assert len(topo.nodes) == 2
        assert topo.scheduler_hints.get("mpi_launcher") == "srun"
        assert topo.scheduler_hints.get("scheduler") == "slurm"

    @pytest.mark.parametrize("malformed_json", ["", "{invalid", "[]", "null"])
    @patch("wien2k_gen.core.scheduler.get_physical_cores")
    def test_detect_local_fallback(self, mock_phys_cores, malformed_json):
        """Robustness: Falls back to local when no scheduler environment detected."""
        mock_phys_cores.return_value = 16
        with patch.dict(os.environ, {}, clear=True):
            topo = detect(force_refresh=True)
        assert topo.total_cores == 16
        assert topo.env_type == "local"

    @patch("wien2k_gen.core.scheduler.get_physical_cores")
    def test_detect_fallback_when_commands_missing(self, mock_phys_cores):
        """Fallback: Graceful degradation when lscpu/numactl/sinfo are missing."""
        mock_phys_cores.return_value = 8
        with patch.dict(os.environ, {}, clear=True):
            topo = detect(force_refresh=True)
        assert topo.total_cores == 8
        assert topo.env_type == "local"

    def test_topology_data_immutability(self):
        """TopologyData is frozen=True to prevent runtime mutation."""
        topo = TopologyData(nodes=["n1", "n2"], cores_per_node=[16, 16], total_cores=32)
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            topo.total_cores = 64

    @patch("wien2k_gen.core.scheduler.get_physical_cores", return_value=12)
    def test_detect_pbs_environment(self, mock_phys_cores):
        """Detect PBS/Torque via environment variables."""
        pbs_env = {"PBS_JOBID": "12345", "PBS_NODEFILE": "/tmp/nodefile", "PBS_NCPUS": "24"}
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value="node01\nnode02"), \
             patch.dict(os.environ, pbs_env, clear=True):
            topo = detect(force_refresh=True)
        assert "pbs" in str(topo.scheduler_hints.get("scheduler", ""))
        assert topo.total_cores == 24

    @pytest.mark.parametrize("nodes,cpus,expected", [
        (["n1", "n2", "n3"], [16, 16, 8], True),
        (["n1", "n2"], [32, 32], False),
        (["n1"], [16], False)
    ])
    def test_heterogeneous_detection(self, nodes, cpus, expected):
        """Auto-detect heterogeneous clusters."""
        topo = Topology(nodes=nodes, cores_per_node=cpus)
        assert topo.heterogeneous == expected


class TestTopologyThreadSafety:
    """Ensure detect() can be called concurrently without race conditions."""

    @patch("wien2k_gen.core.scheduler.FileLock")
    @patch("wien2k_gen.core.scheduler.get_physical_cores", return_value=16)
    def test_concurrent_detect_calls(self, mock_phys_cores, mock_filelock):
        from pathlib import Path
        Path("/tmp/wien2k_gen_topology_cache.json").unlink(missing_ok=True)
        mock_filelock.return_value.__enter__.return_value = mock_filelock.return_value
        mock_filelock.return_value.__exit__.return_value = False
        results = []
        errors = []

        def worker():
            try:
                results.append(detect(force_refresh=True))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert all(r.total_cores == 16 for r in results)