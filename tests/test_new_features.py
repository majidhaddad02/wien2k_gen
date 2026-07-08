"""Tests for new features: SGE, MPICH/MVAPICH, --reserve-os-cores, system type, CPU gen."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from wien2k_gen.core.scheduler import _detect_sge
from wien2k_gen.core.topology import Topology


# ============================================================
# SGE / GridEngine detection
# ============================================================

def test_sge_no_env_returns_none() -> None:
    with patch.dict(os.environ, {}, clear=True):
        result = _detect_sge()
        assert result is None


def test_sge_with_env_no_hostfile() -> None:
    env = {"SGE_JOB_ID": "12345", "NSLOTS": "64", "NHOSTS": "4"}
    with patch.dict(os.environ, env, clear=True):
        result = _detect_sge()
        assert result is not None
        assert result["scheduler"] == "sge"
        assert result["total_cores"] == 64
        assert len(result["nodes"]) == 4


def test_sge_with_hostfile() -> None:
    hostfile_content = """node01  16  all.q  0-15
node01  16  all.q  16-31
node02  16  all.q  0-15
"""
    with tempfile.TemporaryDirectory() as d:
        hostfile = Path(d) / "pe_hostfile"
        hostfile.write_text(hostfile_content)
        env = {"SGE_JOB_ID": "12345", "PE_HOSTFILE": str(hostfile)}
        with patch.dict(os.environ, env, clear=True):
            result = _detect_sge()
            assert result is not None
            assert result["scheduler"] == "sge"
            assert result["total_cores"] == 48
            assert "node01" in result["nodes"]
            assert "node02" in result["nodes"]


# ============================================================
# MPICH / MVAPICH binding hints
# ============================================================

def test_mpich_binding_hints_present() -> None:
    topo = Topology(nodes=["n1"], cores_per_node=[16])
    hints = topo.get_mpi_binding_hints()
    assert "mpich" in hints
    assert "-bind-to" in hints["mpich"]


def test_mvapich_binding_hints_present() -> None:
    topo = Topology(nodes=["n1"], cores_per_node=[16])
    hints = topo.get_mpi_binding_hints()
    assert "mvapich" in hints
    assert "MV2_" in hints["mvapich"]


def test_mpich_hints_vary_by_topology() -> None:
    topo_single = Topology(nodes=["n1"], cores_per_node=[16])
    hints = topo_single.get_mpi_binding_hints()
    assert "mpich" in hints
    assert "-bind-to" in hints["mpich"]


# ============================================================
# CPU generation detection
# ============================================================

@patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
def test_cpu_gen_xeon_sapphire_rapids(mock_run) -> None:
    mock_run.return_value = "Model name: Intel(R) Xeon(R) Platinum 8480+"
    from wien2k_gen.core.hardware import SysFSHardwareInfo
    provider = SysFSHardwareInfo()
    result = provider.get_cpu_generation()
    assert "SapphireRapids" in result


@patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
def test_cpu_gen_epyc_genoa(mock_run) -> None:
    mock_run.return_value = "Model name: AMD EPYC 9654 96-Core Processor"
    from wien2k_gen.core.hardware import SysFSHardwareInfo
    provider = SysFSHardwareInfo()
    result = provider.get_cpu_generation()
    assert "Genoa" in result


@patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
def test_cpu_gen_unknown(mock_run) -> None:
    mock_run.return_value = "Model name: Some Unknown CPU"
    from wien2k_gen.core.hardware import SysFSHardwareInfo
    provider = SysFSHardwareInfo()
    result = provider.get_cpu_generation()
    assert result == "unknown"


# ============================================================
# System type detection
# ============================================================

@patch("wien2k_gen.core.hardware.SysFSHardwareInfo.get_physical_cores")
@patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
def test_system_type_cluster(mock_run, mock_cores) -> None:
    mock_cores.return_value = 128
    mock_run.return_value = ""
    env = {"SLURM_JOB_ID": "12345"}
    with patch.dict(os.environ, env, clear=True):
        from wien2k_gen.core.hardware import SysFSHardwareInfo
        provider = SysFSHardwareInfo()
        result = provider.get_system_type()
        assert result == "cluster"


@patch("wien2k_gen.core.hardware.SysFSHardwareInfo.get_physical_cores")
@patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
def test_system_type_workstation(mock_run, mock_cores) -> None:
    mock_cores.return_value = 16
    mock_run.return_value = ""
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(Path, "exists", return_value=False):
            from wien2k_gen.core.hardware import SysFSHardwareInfo
            provider = SysFSHardwareInfo()
            result = provider.get_system_type()
            assert result == "workstation"


@patch("wien2k_gen.core.hardware.SysFSHardwareInfo.get_physical_cores")
@patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
def test_system_type_compute_node(mock_run, mock_cores) -> None:
    mock_cores.return_value = 64
    mock_run.return_value = ""
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(Path, "exists", return_value=False):
            from wien2k_gen.core.hardware import SysFSHardwareInfo
            provider = SysFSHardwareInfo()
            result = provider.get_system_type()
            assert result == "compute_node"
