"""
Central Pytest Configuration & Shared Fixtures.
Provides reusable mocks, temporary directories, environment patches,
and deterministic data generators for all test modules.
"""

import os
from typing import Any
from unittest.mock import patch

import pytest

from forge.types import TopologyData

# =============================================================================
# Core Data Fixtures
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
def mock_hardware_profile() -> dict[str, Any]:
    return {
        "cpu_arch": "x86_64", "physical_cores": 64, "logical_cores": 128,
        "sockets": 2, "cores_per_socket": 32, "memory_gb": 256.0,
        "memory_bandwidth_gb_s": 200.0, "peak_fp64_gflops": 800.0,
        "numa_nodes": 2, "vector_isa": "avx512", "interconnect_type": "infiniband",
        "interconnect_provider": "mlx", "scratch_fs": "tmpfs",
        "elpa_available": True, "mkl_available": True
    }


# =============================================================================
# Environment & Path Mocks
# =============================================================================

@pytest.fixture
def clean_env():
    """Isolate tests from host environment variables."""
    saved = os.environ.copy()
    preserve = {"PATH", "HOME", "USER", "LOGNAME"}
    clean = {k: v for k, v in saved.items() if k in preserve or k.startswith("PYTEST")}
    os.environ.clear()
    os.environ.update(clean)
    try:
        yield os.environ
    finally:
        os.environ.clear()
        os.environ.update(saved)

@pytest.fixture
def temp_config_dir(tmp_path):
    """Provide a clean, isolated directory for config/cache testing."""
    cfg_dir = tmp_path / ".config" / "forge"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir


# =============================================================================
# I/O & Subprocess Mocks
# =============================================================================

@pytest.fixture
def mock_subprocess_run():
    """Replace subprocess.run with a controllable mock."""
    with patch("subprocess.run") as mock_run:
        yield mock_run

@pytest.fixture
def mock_atomic_write(tmp_path):
    """Bypass actual filesystem writes during tests."""
    with patch("forge.utils.atomic_write.atomic_write") as mock_write:
        mock_write.side_effect = lambda path, content, **kw: True
        yield mock_write


# =============================================================================
# Pytest Hooks & Config
# =============================================================================

def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests requiring external CLI/tools")
    config.addinivalue_line("markers", "hardware: marks tests simulating real cluster behavior")