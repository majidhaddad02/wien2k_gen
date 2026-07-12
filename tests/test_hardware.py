"""
Production-Grade Tests for core.hardware Module.
Tests hardware detection functions using mocked /proc and subprocess calls.
Uses SysFSHardwareInfo for direct provider testing and @patch for system isolation.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from wien2k_gen.core.hardware import (
    HardwareInfoProvider,
    SysFSHardwareInfo,
    get_logical_cores,
    get_provider,
    parse_cpu_list,
    parse_memory_string,
    set_provider,
)


@pytest.fixture
def provider():
    return SysFSHardwareInfo()


@pytest.fixture
def mock_cpuinfo():
    return (
        "processor\t: 0\n"
        "vendor_id\t: GenuineIntel\n"
        "cpu family\t: 6\n"
        "model\t\t: 85\n"
        "model name\t: Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz\n"
        "stepping\t: 7\n"
        "physical id\t: 0\n"
        "core id\t\t: 0\n"
        "cpu MHz\t\t: 2999.998\n\n"
        "processor\t: 1\n"
        "physical id\t: 0\n"
        "core id\t\t: 1\n"
        "cpu MHz\t\t: 2999.998\n\n"
        "processor\t: 2\n"
        "physical id\t: 1\n"
        "core id\t\t: 0\n"
        "cpu MHz\t\t: 2999.998\n\n"
        "processor\t: 3\n"
        "physical id\t: 1\n"
        "core id\t\t: 1\n"
        "cpu MHz\t\t: 2999.998\n\n"
    )


@pytest.fixture
def mock_meminfo():
    return "MemTotal:       131934756 kB\nMemFree:        65432100 kB\n"


@pytest.fixture
def mock_lscpu_json():
    return json.dumps({
        "lscpu": [
            {"field": "Architecture:", "data": "x86_64"},
            {"field": "CPU(s):", "data": "96"},
            {"field": "Thread(s) per core:", "data": "2"},
            {"field": "Core(s) per socket:", "data": "24"},
            {"field": "Socket(s):", "data": "2"},
            {"field": "NUMA node(s):", "data": "2"},
            {"field": "Model name:", "data": "Intel(R) Xeon(R) Gold 6248R"},
            {"field": "Flags:", "data": "fpu avx avx2 avx512f avx512dq sse4_1 sse4_2"},
        ]
    })


@pytest.fixture
def mock_lscpu_csv():
    return (
        "# CPU,SOCKET,CORE\n"
        "0,0,0\n1,0,1\n2,0,2\n3,0,3\n"
        "4,1,0\n5,1,1\n6,1,2\n7,1,3\n"
    )


# =============================================================================
# Provider Interface Tests
# =============================================================================

class TestHardwareInfoProvider:
    """Test that SysFSHardwareInfo correctly implements HardwareInfoProvider."""

    def test_is_instance_of_abc(self, provider):
        assert isinstance(provider, HardwareInfoProvider)

    def test_has_all_abstract_methods(self, provider):
        methods = [
            "get_logical_cores", "get_physical_cores", "is_hyperthreading_active",
            "get_vector_isa_and_width", "get_fma_units_per_core",
            "calculate_peak_fp64_gflops", "get_cpu_governor",
            "get_cpu_frequency_info", "get_job_memory_limit_mb",
            "get_numa_topology_detailed", "get_cache_topology",
            "get_total_mem_kb", "get_scratch_filesystem_type",
            "get_interconnect_info", "get_cpu_architecture",
            "get_memory_bandwidth_gb_s", "is_containerized",
            "check_elpa_available", "check_mkl_available",
            "get_hardware_profile",
        ]
        for method in methods:
            assert hasattr(provider, method), f"Missing method: {method}"

    def test_set_provider_swaps_instance(self):
        original = get_provider()
        mock_provider = MagicMock(spec=HardwareInfoProvider)
        mock_provider.get_logical_cores.return_value = 1000
        try:
            set_provider(mock_provider)
            assert get_logical_cores() == 1000
        finally:
            set_provider(original)


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestParseCpuList:
    def test_single_value(self):
        assert parse_cpu_list("5") == [5]

    def test_range(self):
        assert parse_cpu_list("0-3") == [0, 1, 2, 3]

    def test_comma_separated(self):
        assert parse_cpu_list("0-1,4,6-7") == [0, 1, 4, 6, 7]

    def test_empty_string(self):
        assert parse_cpu_list("") == []

    def test_whitespace_only(self):
        assert parse_cpu_list("   ") == []

    def test_invalid_input(self):
        assert parse_cpu_list("abc") == []


class TestParseMemoryString:
    def test_gigabytes(self):
        assert parse_memory_string("10G") == 10240

    def test_megabytes(self):
        assert parse_memory_string("1024M") == 1024

    def test_kilobytes(self):
        mb = parse_memory_string("1048576K")
        assert mb == 1024

    def test_terabytes(self):
        mb = parse_memory_string("1T")
        assert mb == 1024 * 1024

    def test_no_unit(self):
        assert parse_memory_string("512") == 512

    def test_large_raw_value(self):
        mb = parse_memory_string("1048576K")
        assert mb == 1024

    def test_empty_string(self):
        assert parse_memory_string("") is None

    def test_with_trailing_b(self):
        assert parse_memory_string("4GB") == 4096


# =============================================================================
# Core Detection Tests with Mocks
# =============================================================================

class TestPhysicalCores:
    @patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
    def test_lscpu_json_method(self, mock_run, provider, mock_lscpu_json):
        mock_run.return_value = mock_lscpu_json
        assert provider.get_physical_cores() == 48

    @patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
    def test_lscpu_csv_fallback(self, mock_run, provider, mock_lscpu_csv):
        mock_run.side_effect = [None, mock_lscpu_csv]
        assert provider.get_physical_cores() == 8

    @patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe", return_value=None)
    def test_proc_cpuinfo_fallback(self, mock_run, provider, mock_cpuinfo):
        with patch("builtins.open", mock_open(read_data=mock_cpuinfo)):
            cores = provider.get_physical_cores()
            assert cores == 4

    @patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe", return_value=None)
    def test_ultimate_fallback(self, mock_run, provider):
        with patch("builtins.open", side_effect=FileNotFoundError):
            cores = provider.get_physical_cores()
            assert cores == provider.get_logical_cores()


class TestTotalMemory:
    @patch("pathlib.Path.read_text")
    def test_meminfo_parsing(self, mock_read, provider, mock_meminfo):
        mock_read.return_value = mock_meminfo
        kb = provider.get_total_mem_kb()
        assert kb == 131934756

    def test_meminfo_fallback(self, provider):
        with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
            kb = provider.get_total_mem_kb()
            assert kb == 4 * 1024 * 1024


class TestELPAAvailability:
    def test_elpa_found(self, provider):
        with patch("pathlib.Path.exists", return_value=True):
            assert provider.check_elpa_available() is True

    def test_elpa_not_found(self, provider):
        with patch("pathlib.Path.exists", return_value=False):
            assert provider.check_elpa_available() is False

    def test_elpa_with_wienroot(self, provider):
        with patch.dict(os.environ, {"WIENROOT": "/custom/path"}), patch("pathlib.Path.exists", return_value=True):
                assert provider.check_elpa_available() is True


class TestScratchFilesystem:
    @patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
    def test_lustre_detection(self, mock_run, provider):
        mock_run.return_value = (
            "Filesystem     Type  1K-blocks      Used Available Use% Mounted on\n"
            "/scratch       lustre 500000000 300000000 200000000  60% /scratch"
        )
        assert provider.get_scratch_filesystem_type() == "lustre"

    @patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
    def test_tmpfs_detection(self, mock_run, provider):
        mock_run.return_value = (
            "Filesystem     Type  1K-blocks      Used Available Use% Mounted on\n"
            "/dev/shm       tmpfs  64000000         0  64000000   0% /dev/shm"
        )
        assert provider.get_scratch_filesystem_type() == "tmpfs"

    @patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe", return_value=None)
    def test_df_failure_fallback(self, mock_run, provider):
        assert provider.get_scratch_filesystem_type() == "unknown"


class TestIsaDetection:
    @patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
    def test_avx512_detection(self, mock_run, provider):
        mock_run.return_value = "Flags: fpu vme de pse avx512f avx512dq avx512bw"
        info = provider.get_vector_isa_and_width()
        assert info["isa"] == "avx512"
        assert info["width_bits"] == 512

    @patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
    def test_avx2_detection(self, mock_run, provider):
        mock_run.return_value = "Flags: fpu avx2 sse4_1 sse4_2"
        info = provider.get_vector_isa_and_width()
        assert info["isa"] == "avx2"
        assert info["width_bits"] == 256

    @patch("wien2k_gen.core.hardware.SysFSHardwareInfo._run_cmd_safe")
    def test_scalar_fallback(self, mock_run, provider):
        mock_run.return_value = "Flags: fpu mmx"
        info = provider.get_vector_isa_and_width()
        assert info["isa"] == "scalar"
        assert info["width_bits"] == 64


class TestHardwareProfile:
    @patch.object(SysFSHardwareInfo, "_run_cmd_safe", return_value=None)
    @patch.object(SysFSHardwareInfo, "get_physical_cores", return_value=8)
    @patch.object(SysFSHardwareInfo, "get_logical_cores", return_value=16)
    @patch.object(SysFSHardwareInfo, "get_cpu_architecture", return_value="xeon")
    @patch.object(SysFSHardwareInfo, "get_vector_isa_and_width", return_value={"isa": "avx2", "width_bits": 256})
    @patch.object(SysFSHardwareInfo, "get_fma_units_per_core", return_value=2)
    @patch.object(SysFSHardwareInfo, "get_cpu_frequency_info", return_value={"min": 1000.0, "max": 3000.0, "current": 2000.0, "base": 2500.0})
    @patch.object(SysFSHardwareInfo, "get_total_mem_kb", return_value=67108864)
    @patch.object(SysFSHardwareInfo, "get_job_memory_limit_mb", return_value=None)
    @patch.object(SysFSHardwareInfo, "get_numa_topology_detailed", return_value=[])
    @patch.object(SysFSHardwareInfo, "get_cache_topology", return_value=[])
    @patch.object(SysFSHardwareInfo, "get_interconnect_info", return_value={"type": "unknown", "provider": "unknown", "speed_gbps": 10.0, "latency_ns": 1000.0, "numa_aware": False})
    @patch.object(SysFSHardwareInfo, "get_scratch_filesystem_type", return_value="tmpfs")
    @patch.object(SysFSHardwareInfo, "check_elpa_available", return_value=True)
    @patch.object(SysFSHardwareInfo, "check_mkl_available", return_value=True)
    @patch.object(SysFSHardwareInfo, "is_containerized", return_value=False)
    @patch.object(SysFSHardwareInfo, "get_memory_bandwidth_gb_s", return_value=168.0)
    def test_hardware_profile_assembly(self, mock_bandwidth, mock_containerized,
                                        mock_mkl, mock_elpa, mock_scratch,
                                        mock_interconnect, mock_cache, mock_numa,
                                        mock_mem_limit, mock_mem_total, mock_freq,
                                        mock_fma, mock_isa, mock_arch,
                                        mock_logical, mock_physical, mock_cmd,
                                        provider):
        profile = provider.get_hardware_profile()
        assert profile["physical_cores"] == 8
        assert profile["logical_cores"] == 16
        assert profile["hyperthreading"] is True
        assert profile["memory_total_gb"] == 64.0
        assert profile["cpu_arch"] == "xeon"
        assert profile["vector_isa"] == "avx2"
        assert profile["elpa_available"] is True
        assert profile["mkl_available"] is True
        assert isinstance(profile["peak_fp64_gflops"], float)

    @patch.object(SysFSHardwareInfo, "get_numa_topology_detailed", return_value=[{"node_id": 0, "cpus": "0-3", "cpu_ids": [0, 1, 2, 3], "mem_kb": 33554432, "cores": 4, "distance": {0: 10}}])
    def test_get_numa_node_count(self, mock_numa, provider):
        assert provider.get_numa_node_count() == 1


class TestContainerDetection:
    def test_not_containerized(self, provider):
        with patch.object(Path, "exists", return_value=False), patch.dict(os.environ, {}, clear=True):
                assert provider.is_containerized() is False

    def test_docker_containerized(self, provider):
        with patch("pathlib.Path.exists", return_value=True):
            assert provider.is_containerized() is True

    def test_singularity_containerized(self, provider):
        with patch.object(Path, "exists", return_value=False), patch.dict(os.environ, {"SINGULARITY_CONTAINER": "1"}):
            assert provider.is_containerized() is True
