"""
Integration Tests for FORGE.
Verifies end-to-end connectivity between core modules, CLI, configuration,
optimization engine, and type system without requiring hardware or cluster access.
"""

import json
from typing import Any
from unittest.mock import patch

import pytest

from forge.cli import create_parser
from forge.config import ConfigManager, load_config
from forge.core.scheduler import apply_max_cores
from forge.core.topology import Topology
from forge.optimizer.advisor import suggest_optimal_resources

# Project imports
from forge.types import ExecutionMode, PipelineResult, ResourceSuggestion, TopologyData

# =============================================================================
# Fixtures for Integration
# =============================================================================

@pytest.fixture
def mock_slurm_env():
    """Provide a simulated SLURM environment."""
    return {
        "SLURM_JOB_ID": "12345",
        "SLURM_NNODES": "2",
        "SLURM_TASKS_PER_NODE": "32(x2)",
        "SLURM_JOB_NODELIST": "node[01-02]"
    }

@pytest.fixture
def sample_topology() -> TopologyData:
    return TopologyData(
        nodes=["node1", "node2"],
        cores_per_node=[32, 32],
        env_type="slurm",
        total_cores=64,
        scheduler_hints={"mpi_launcher": "srun", "numa_aware": True},
        heterogeneous=False
    )

@pytest.fixture
def sample_hardware_profile() -> dict[str, Any]:
    return {
        "cpu_arch": "x86_64", "physical_cores": 64, "memory_gb": 256.0,
        "peak_fp64_gflops": 800.0, "numa_nodes": 2, "interconnect_type": "infiniband",
        "elpa_available": True, "mkl_available": True
    }


# =============================================================================
# Integration Tests
# =============================================================================

class TestTypeSystemSerialization:
    """Verify type models are JSON-serializable and preserve data."""

    def test_resource_suggestion_roundtrip(self, sample_topology):
        suggestion = ResourceSuggestion(
            mode=ExecutionMode.HYBRID,
            recommended_total_cores=32,
            omp_threads_per_rank=4,
            mpi_ranks_per_node=8,
            warnings=["Test warning"],
            reason="Unit test"
        )
        
        # Serialize
        data = suggestion.to_dict()
        json_str = json.dumps(data, default=str)
        
        # Deserialize
        loaded = json.loads(json_str)
        assert loaded["mode"] == "hybrid"
        assert loaded["recommended_total_cores"] == 32
        assert "Test warning" in loaded["warnings"]

    def test_pipeline_result_serialization(self):
        result = PipelineResult(
            success=True,
            config_path="/tmp/.machines",
            config_content="lapw1: node01: 16",
            warnings=["Low memory"],
            metadata={"elapsed": 0.5}
        )
        assert result.is_valid()
        data = result.to_dict()
        assert data["success"] is True
        assert data["config_path"] == "/tmp/.machines"


class TestCoreToOptimizerFlow:
    """Verify data flows correctly from Scheduler to Advisor."""

    @patch("forge.optimizer.advisor.get_total_mem_kb", return_value=256 * 1024 * 1024)
    @patch("forge.optimizer.advisor.get_physical_cores", return_value=64)
    @patch("forge.optimizer.advisor.get_memory_bandwidth_gb_s", return_value=200.0)
    @patch("forge.optimizer.advisor.calculate_peak_fp64_gflops", return_value=800.0)
    def test_suggestion_generation(self, mock_flops, mock_bw, mock_cores, mock_mem,
                                   sample_hardware_profile):
        topo = Topology(
            nodes=["node1", "node2"],
            cores_per_node=[32, 32],
            env_type="slurm",
            scheduler_hints={"mpi_launcher": "srun", "numa_aware": True},
        )
        suggestion = suggest_optimal_resources(topo)
        data = suggestion.to_dict() if hasattr(suggestion, "to_dict") else suggestion

        assert data["recommended_total_cores"] <= 64
        assert data["mode"] in ["mpi", "hybrid", "kpoint"]
        assert "reason" in data
        assert isinstance(data["warnings"], list)

    def test_apply_max_cores_integrity(self):
        topo = Topology(
            nodes=["node1", "node2"],
            cores_per_node=[32, 32],
            env_type="slurm",
        )
        # Test limiting cores
        limited = apply_max_cores(topo, 20)
        assert sum(limited.cores_per_node) == 20
        
        # Test expanding (should return original or capped)
        expanded = apply_max_cores(topo, 128)
        assert expanded.total_cores == 64


class TestConfigurationManager:
    """Test ConfigManager persistence and CLI integration."""

    def test_config_load_and_save(self, tmp_path):
        config_path = tmp_path / "config.json"
        
        # Load with overrides
        cfg = load_config(
            file_path=None,
            cli_override={
                "wienroot": "/mock/wien2k",
                "backend": "qe"
            }
        )
        
        assert cfg.wienroot == "/mock/wien2k"
        assert cfg.backend == "qe"
        
        # Save
        assert ConfigManager.instance().save(path=config_path)
        
        # Verify content
        with open(config_path) as f:
            data = json.load(f)
        assert data["backend"] == "qe"

    def test_env_var_precedence(self, monkeypatch):
        monkeypatch.setenv("WIENROOT", "/env/path")
        monkeypatch.setenv("WIEN2K_BACKEND", "vasp")
        
        cfg = load_config()
        assert cfg.wienroot == "/env/path"
        assert cfg.backend == "vasp"


class TestCLIArgumentRouting:
    """Verify CLI arguments map to correct modules."""

    def test_parser_subcommands(self):
        parser = create_parser()
        
        # Test 'generate' command
        args = parser.parse_args(["generate", "--cores", "64", "--mode", "hybrid"])
        assert args.command == "generate"
        assert args.cores == 64
        assert args.mode == "hybrid"

    def test_parser_validation(self, capsys):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["generate", "--mode", "invalid_mode"])
        
        captured = capsys.readouterr()
        assert "invalid" in captured.err.lower()


class TestEndToEndPipelineSimulation:
    """Simulate the full lifecycle: Detect -> Suggest -> Config -> Result."""

    def test_mocked_pipeline(self, sample_hardware_profile):
        topo = Topology(
            nodes=["node1", "node2"],
            cores_per_node=[16, 16],
            env_type="slurm",
            scheduler_hints={"mpi_launcher": "srun", "numa_aware": True},
        )
        assert topo.total_cores == 32

        # Suggest
        with patch("forge.optimizer.advisor.get_total_mem_kb", return_value=256 * 1024 * 1024), \
             patch("forge.optimizer.advisor.get_physical_cores", return_value=32), \
             patch("forge.optimizer.advisor.get_memory_bandwidth_gb_s", return_value=200.0), \
             patch("forge.optimizer.advisor.calculate_peak_fp64_gflops", return_value=800.0):
            suggestion = suggest_optimal_resources(topo)
        data = suggestion.to_dict() if hasattr(suggestion, "to_dict") else suggestion

        # Validate Type Consistency
        assert data["recommended_total_cores"] <= 32
        assert data["recommended_total_cores"] == int(data["recommended_total_cores"])
        assert isinstance(data["mode"], str)
        assert data["mode"] in [m.value for m in ExecutionMode]