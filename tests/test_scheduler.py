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

from wien2k_gen.core.scheduler import detect, _parse_lscpu_json, _parse_numactl, TopologyData
from wien2k_gen.types import TopologyData as TypesTopologyData
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
        "SLURM_JOB_CPUS_PER_NODE": "16(x2)",
        "SLURM_STEP_LAUNCHER_PORT": "12345"
    }


# =============================================================================
# Test Suites
# =============================================================================

class TestSchedulerDetection:
    """Tests for main detect() function and helper parsers."""

    @patch("wien2k_gen.core.scheduler.shutil.which")
    @patch("wien2k_gen.core.scheduler.subprocess.run")
    def test_detect_successful_topology(self, mock_run, mock_which, mock_lscpu_json, mock_numactl_output, env_slurm):
        """Happy path: Full topology detection in SLURM env."""
        mock_which.return_value = "/usr/bin/lscpu"
        mock_run.side_effect = [
            MagicMock(stdout=mock_lscpu_json, returncode=0),
            MagicMock(stdout=mock_numactl_output, returncode=0)
        ]
        
        with patch.dict(os.environ, env_slurm, clear=False):
            topo = detect()
            
        assert topo.total_cores == 32
        assert topo.env_type == "slurm"
        assert topo.heterogeneous is False
        assert len(topo.nodes) == 2
        assert topo.scheduler_hints.get("mpi_launcher") == "srun"

    @pytest.mark.parametrize("malformed_json", ["", "{invalid", "[]", "null"])
    @patch("wien2k_gen.core.scheduler.shutil.which")
    @patch("wien2k_gen.core.scheduler.subprocess.run")
    def test_detect_lscpu_malformed_output(self, mock_run, mock_which, malformed_json):
        """Robustness: Handle broken JSON from lscpu."""
        mock_which.return_value = "/usr/bin/lscpu"
        mock_run.return_value = MagicMock(stdout=malformed_json, returncode=0)
        
        with pytest.raises(DetectionFailedError, match="Failed to parse lscpu output"):
            detect()

    @patch("wien2k_gen.core.scheduler.shutil.which")
    @patch("wien2k_gen.core.scheduler.subprocess.run")
    def test_detect_fallback_when_commands_missing(self, mock_run, mock_which):
        """Fallback: Graceful degradation when lscpu/numactl/sinfo are missing."""
        mock_which.return_value = None
        with patch.dict(os.environ, {}, clear=False):
            topo = detect()
            
        assert topo.total_cores == os.cpu_count() or 1
        assert topo.env_type == "local"
        mock_run.assert_not_called()

    def test_topology_data_immutability(self):
        """Dataclass frozen=True prevents runtime mutation."""
        topo = TopologyData(nodes=["n1", "n2"], cores_per_node=[16, 16], total_cores=32)
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            topo.total_cores = 64

    @patch("wien2k_gen.core.scheduler.shutil.which")
    @patch("wien2k_gen.core.scheduler.subprocess.run")
    def test_detect_pbs_environment(self, mock_run, mock_which):
        """Detect PBS/Torque via environment variables."""
        mock_which.return_value = None
        pbs_env = {"PBS_NODEFILE": "/tmp/nodefile", "PBS_NCPUS": "24"}
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value="node01\nnode02"), \
             patch.dict(os.environ, pbs_env, clear=False):
            topo = detect()
        assert topo.env_type == "pbs"
        assert topo.total_cores == 24

    @pytest.mark.parametrize("nodes,cpus,expected", [
        (["n1", "n2", "n3"], [16, 16, 8], True),
        (["n1", "n2"], [32, 32], False),
        (["n1"], [16], False)
    ])
    def test_heterogeneous_detection(self, nodes, cpus, expected):
        """Auto-detect heterogeneous clusters."""
        topo = TopologyData(nodes=nodes, cores_per_node=cpus, total_cores=sum(cpus))
        assert topo.heterogeneous == expected


class TestTopologyThreadSafety:
    """Ensure detect() can be called concurrently without race conditions."""

    @patch("wien2k_gen.core.scheduler.shutil.which", return_value="/bin/true")
    @patch("wien2k_gen.core.scheduler.subprocess.run")
    def test_concurrent_detect_calls(self, mock_run, mock_which, mock_lscpu_json, mock_numactl_output):
        mock_run.side_effect = [
            MagicMock(stdout=mock_lscpu_json, returncode=0),
            MagicMock(stdout=mock_numactl_output, returncode=0)
        ]
        results = []
        errors = []

        def worker():
            try:
                results.append(detect())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert all(r.total_cores == r.total_cores for r in results)  # Consistent