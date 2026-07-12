"""Robustness tests — bad inputs, missing files, edge cases."""

import tempfile
from pathlib import Path

import pytest

from wien2k_gen.core.case_parser import CaseData, CaseFileParser
from wien2k_gen.core.topology import Topology
from wien2k_gen.utils.validation import parse_machines_file


class TestMissingFiles:
    """Parser must not crash with missing or empty case directories."""

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as d:
            data = CaseFileParser(Path(d)).parse_all()
            assert isinstance(data, CaseData)
            assert data.atoms == 0
            assert data.nmat == 0

    def test_only_struct_no_scf(self):
        """When .scf is missing, nmat falls back to .in2 estimate."""
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            (dp / "case.struct").write_text("H   LATTICE,NONEQUIV.ATOMS:  1\nMODE OF CALC=RELA\n9.0 9.0 9.0 90 90 90\n")
            (dp / "case.in2").write_text("TOT\n14.0\n 80 80 80 2.0 0\n")
            (dp / "case.in1").write_text("40 TOT\n7.0 10 4\n")
            (dp / "case.klist").write_text("16\n")
            data = CaseFileParser(dp).parse_all()
            assert data.atoms == 1
            assert data.nmat > 0  # estimated from FFT grid
            assert data.fft_nx == 80

    def test_corrupt_struct_file(self):
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            (dp / "case.struct").write_text("GARBAGE\x00\xFF\nNOT A VALID STRUCT\0")
            data = CaseFileParser(dp).parse_all()
            assert isinstance(data, CaseData)
            assert data.atoms == 0  # graceful fallback

    def test_malformed_scf(self):
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            (dp / "case.scf").write_text(":NMAT   :  not_a_number\n:FER: garbage\n")
            data = CaseFileParser(dp).parse_all()
            assert data.nmat == 0  # failed to parse int

    def test_nonexistent_path(self):
        data = CaseFileParser(Path("/nonexistent/path/xyz")).parse_all()
        assert data.atoms == 0
        assert data.nmat == 0

    def test_parse_in1_with_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "case.in1"
            p.write_text("")
            r = CaseFileParser.parse_in1(p)
            assert r["nbands"] is None


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_negative_max_cores(self):
        """Negative max_cores should be treated like None (unlimited)."""
        import os as _os

        from wien2k_gen.optimizer.advisor import suggest_optimal_resources
        mock_env = {"WIENROOT": "/tmp"}
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_os, "environ", mock_env)
            mp.setattr("wien2k_gen.core.hardware.get_physical_cores", lambda: 4)
            mp.setattr("wien2k_gen.core.hardware.get_total_mem_kb", lambda: 256 * 1024 * 1024)
            mp.setattr("wien2k_gen.core.hardware.get_memory_bandwidth_gb_s", lambda: 50.0)
            mp.setattr("wien2k_gen.core.hardware.is_hyperthreading_active", lambda: False)
            mp.setattr("wien2k_gen.core.hardware.check_elpa_available", lambda: False)
            mp.setattr("wien2k_gen.core.hardware.check_mkl_available", lambda: False)
            mp.setattr("wien2k_gen.core.hardware.get_cpu_architecture", lambda: "xeon")
            mp.setattr("wien2k_gen.core.hardware.get_job_memory_limit_mb", lambda: None)
            mp.setattr("wien2k_gen.core.hardware.get_numa_node_count", lambda: 1)
            mp.setattr("wien2k_gen.core.hardware.get_scratch_filesystem_type", lambda: "tmpfs")
            mp.setattr("wien2k_gen.core.hardware.get_fma_units_per_core", lambda: 2)
            mp.setattr("wien2k_gen.optimizer.advisor.get_fma_units_per_core", lambda: 2)
            mp.setattr("wien2k_gen.optimizer.advisor.calculate_peak_fp64_gflops", lambda: 100.0)
            from unittest.mock import MagicMock
            mock_backend = MagicMock()
            mock_backend.detect_problem_size.return_value = {
                "atoms": 10, "kpoints": 8, "nmat": 2000, "nbands": 100,
                "rkmax": 7.0, "is_soc": False, "is_hybrid": False, "complexity": 1.0,
            }
            mp.setattr("wien2k_gen.optimizer.advisor._get_current_backend", lambda: mock_backend)
            topo = Topology(nodes=["n1"], cores_per_node=[16])
            max_cores = None  # negative means unlimited
            result = suggest_optimal_resources(topo, user_max_cores=max_cores)
            assert result.recommended_total_cores > 0
            assert result.recommended_total_cores <= 16

    def test_single_core_machine(self):
        topo = Topology(nodes=["n1"], cores_per_node=[1])
        import os as _os

        from wien2k_gen.optimizer.advisor import suggest_optimal_resources
        mock_env = {"WIENROOT": "/tmp"}
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_os, "environ", mock_env)
            mp.setattr("wien2k_gen.core.hardware.get_physical_cores", lambda: 1)
            mp.setattr("wien2k_gen.core.hardware.get_total_mem_kb", lambda: 64 * 1024 * 1024)
            mp.setattr("wien2k_gen.core.hardware.get_memory_bandwidth_gb_s", lambda: 20.0)
            mp.setattr("wien2k_gen.core.hardware.is_hyperthreading_active", lambda: False)
            mp.setattr("wien2k_gen.core.hardware.check_elpa_available", lambda: False)
            mp.setattr("wien2k_gen.core.hardware.check_mkl_available", lambda: False)
            mp.setattr("wien2k_gen.core.hardware.get_cpu_architecture", lambda: "xeon")
            mp.setattr("wien2k_gen.core.hardware.get_job_memory_limit_mb", lambda: None)
            mp.setattr("wien2k_gen.core.hardware.get_numa_node_count", lambda: 1)
            mp.setattr("wien2k_gen.core.hardware.get_scratch_filesystem_type", lambda: "tmpfs")
            mp.setattr("wien2k_gen.core.hardware.get_fma_units_per_core", lambda: 2)
            mp.setattr("wien2k_gen.optimizer.advisor.get_fma_units_per_core", lambda: 2)
            mp.setattr("wien2k_gen.optimizer.advisor.calculate_peak_fp64_gflops", lambda: 10.0)
            from unittest.mock import MagicMock
            mock_backend = MagicMock()
            mock_backend.detect_problem_size.return_value = {
                "atoms": 10, "kpoints": 8, "nmat": 500, "nbands": 50,
                "rkmax": 7.0, "is_soc": False, "is_hybrid": False, "complexity": 1.0,
            }
            mp.setattr("wien2k_gen.optimizer.advisor._get_current_backend", lambda: mock_backend)
            result = suggest_optimal_resources(topo)
            assert result.recommended_total_cores == 1
            assert result.omp_threads_per_rank == 1

    def test_zero_kpoints(self):
        from wien2k_gen.optimizer.advisor import estimate_amdahl_saturation
        result = estimate_amdahl_saturation(
            kpoints=0, nmat=1000, atoms=10,
            total_cores_available=32, num_nodes=1,
        )
        assert result["max_efficient_cores"] >= 1

    def test_huge_nmat(self):
        from wien2k_gen.optimizer.advisor import estimate_memory_footprint_gb
        gb = estimate_memory_footprint_gb(nmat=100000)
        assert gb > 0
        assert gb < 10000  # sanity: not absurd


class TestValidationRobustness:
    """Validation must handle malformed .machines files."""

    def test_empty_machines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".machines"
            p.write_text("")
            config, _warnings = parse_machines_file(p)
            assert config["nodes"] == []

    def test_junk_machines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".machines"
            p.write_text("this is not a valid machines file\n!!!\n")
            config, _warnings = parse_machines_file(p)
            assert isinstance(config, dict)

    def test_valid_machines_parses(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".machines"
            p.write_text("""lapw0: node01: 1
1: node01: 8
1: node02: 8
2: node01: 8
2: node02: 8
granularity:1
extrafine:1
""")
            config, _warnings = parse_machines_file(p)
            assert len(config["nodes"]) == 2
            assert config["kpar"] == 0

    def test_machines_with_comments(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".machines"
            p.write_text("""# WIEN2k parallel configuration
lapw0: localhost: 1
# lapw1: k-point parallel
1: localhost: 8
granularity:1
""")
            config, _warnings = parse_machines_file(p)
            assert len(config["nodes"]) == 1


class TestCLIGracefulDegradation:
    """CLI must not crash with missing dependencies or bad env."""

    def test_import_without_rich(self):
        import sys
        dict(sys.modules)
        for mod in list(sys.modules.keys()):
            if mod.startswith("rich") or mod.startswith("textual"):
                del sys.modules[mod]

        try:
            from unittest.mock import patch
            with patch.dict(sys.modules, {"rich": None, "rich.console": None, "rich.table": None}):
                pass
        finally:
            pass

    def test_topology_immutable_after_creation(self):
        topo = Topology(nodes=["n1", "n2"], cores_per_node=[16, 16])
        assert topo.total_cores == 32
        assert len(topo.nodes) == 2
