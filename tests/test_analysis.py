"""
Tests for ui/analysis.py — SCF Log Parsing & Parallel Scaling Analysis.
Covers WIEN2k/VASP/QE parsers, strong/weak scaling metrics, and report generation.
"""

import pytest

from wien2k_gen.ui.analysis import (
    AnalysisReport,
    calculate_scaling_metrics,
    calculate_weak_scaling_metrics,
    generate_report,
    parse_scf_log,
)

WIEN2K_SCF_CONTENT = """\
WIEN2k 23.1
:ENE  : TOTAL ENERGY = -12345.6789
:DIS  : CHARGE CONVERGENCE = 0.00000001
:DIS  : CHARGE CONVERGENCE = 0.00000002
:DIS  : CHARGE CONVERGENCE = 0.00000003
:DIS  : CHARGE CONVERGENCE = 0.00000004
:DIS  : CHARGE CONVERGENCE = 0.00000005
lapw0  : cpu time : 10.5
lapw1  : cpu time : 45.2
lapw2  : cpu time : 12.3
mixer  : cpu time : 1.1
:CPU  : TOTAL CPU TIME FOR SCF IS 100.0
:REAL : TOTAL WALL TIME FOR SCF IS 90.0
"""

WIEN2K_CONVERGED = """\
WIEN2k 23.1
:ENE  : TOTAL ENERGY = -54321.0
:DIS  : CHARGE CONVERGENCE = 0.00000100
:DIS  : CHARGE CONVERGENCE = 0.00000080
:DIS  : CHARGE CONVERGENCE = 0.00000001
:CPU  : TOTAL CPU TIME FOR SCF IS 50.0
"""

WIEN2K_NOT_CONVERGED = """\
WIEN2k 23.1
:ENE  : TOTAL ENERGY = -10000.0
:DIS  : CHARGE CONVERGENCE = 0.00100000
:DIS  : CHARGE CONVERGENCE = 0.00090000
NOT CONVERGED
QTL-B error detected
:CPU  : TOTAL CPU TIME FOR SCF IS 30.0
"""

VASP_OUTCAR_CONTENT = """\
vasp.6.3.2 21Jan23
FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
--------------
  free  energy   TOTEN  =     -2345.6789 eV
--------------
Iteration      1, Average charge convergence:  0.005000
Iteration      2, Average charge convergence:  0.000500
Iteration      3, Average charge convergence:  0.000010
Iteration      4, Average charge convergence:  0.000008
General timing and accounting info:
 LOOP+:  cpu time     10.50: real time    9.80
"""


class TestParseScfLog:
    def test_auto_detect_wien2k(self, tmp_path):
        log = tmp_path / "case.scf"
        log.write_text(WIEN2K_SCF_CONTENT)
        result = parse_scf_log(str(log))
        assert result["code"] == "wien2k"
        assert result["converged"] is True
        assert result["total_cycles"] == 5
        assert result["final_energy_ry"] == pytest.approx(-12345.6789)
        assert result["stage_timings"]["lapw0"] == 10.5
        assert result["stage_timings"]["lapw1"] == 45.2
        assert result["cpu_time_sec"] == 100.0
        assert result["wall_time_sec"] == 90.0

    def test_converged_within_threshold(self, tmp_path):
        log = tmp_path / "case.scf"
        log.write_text(WIEN2K_CONVERGED)
        result = parse_scf_log(str(log))
        assert result["converged"] is True
        assert result["total_cycles"] == 3

    def test_not_converged(self, tmp_path):
        log = tmp_path / "case.scf"
        log.write_text(WIEN2K_NOT_CONVERGED)
        result = parse_scf_log(str(log))
        assert result["converged"] is False
        assert "SCF did not converge" in str(result["warnings"])

    def test_qtl_b_error_detected(self, tmp_path):
        log = tmp_path / "case.scf"
        log.write_text("WIEN2k 23.1\n" + WIEN2K_SCF_CONTENT + "\nQTL-B error")
        result = parse_scf_log(str(log))
        assert any("QTL-B" in e for e in result["errors"])

    def test_explicit_code_hint_wien2k(self, tmp_path):
        log = tmp_path / "case.scf"
        log.write_text(WIEN2K_SCF_CONTENT)
        result = parse_scf_log(str(log), code_hint="wien2k")
        assert result["code"] == "wien2k"

    def test_auto_detect_vasp(self, tmp_path):
        log = tmp_path / "OUTCAR"
        log.write_text(VASP_OUTCAR_CONTENT)
        result = parse_scf_log(str(log))
        assert result["code"] == "vasp"

    def test_missing_file(self, tmp_path):
        result = parse_scf_log(tmp_path / "nonexistent.log")
        assert result["code"] == "unknown"
        assert "File not found" in str(result["errors"])

    def test_unknown_format(self, tmp_path):
        log = tmp_path / "output.log"
        log.write_text("Some random text with no recognizable markers")
        result = parse_scf_log(str(log))
        assert result["code"] == "unknown"

    def test_qe_detection(self, tmp_path):
        log = tmp_path / "pwscf.out"
        log.write_text("PWSCF output\nQuantum ESPRESSO v7.0\nk-points calculation")
        result = parse_scf_log(str(log))
        assert result["code"] == "qe"


class TestScalingMetrics:
    def test_strong_scaling_excellent(self):
        result = calculate_scaling_metrics(
            base_cores=4, base_time_sec=100.0,
            current_cores=8, current_time_sec=55.0,
        )
        assert result["speedup"] == pytest.approx(1.82, rel=0.1)
        assert result["efficiency_percent"] > 85.0
        assert "Excellent scaling" in result["recommendation"]

    def test_strong_scaling_moderate(self):
        result = calculate_scaling_metrics(
            base_cores=4, base_time_sec=100.0,
            current_cores=8, current_time_sec=80.0,
        )
        assert 40.0 < result["efficiency_percent"] <= 65.0
        assert "Moderate scaling" in result["recommendation"]

    def test_strong_scaling_poor(self):
        result = calculate_scaling_metrics(
            base_cores=4, base_time_sec=100.0,
            current_cores=16, current_time_sec=98.0,
        )
        assert result["efficiency_percent"] < 40.0
        assert "Poor scaling" in result["recommendation"]

    def test_strong_scaling_invalid_timing(self):
        result = calculate_scaling_metrics(
            base_cores=4, base_time_sec=0.0,
            current_cores=8, current_time_sec=50.0,
        )
        assert result["speedup"] == 0.0
        assert result["recommendation"] == "Invalid timing data. Check log parsing or run duration."

    def test_strong_scaling_good(self):
        result = calculate_scaling_metrics(
            base_cores=4, base_time_sec=100.0,
            current_cores=8, current_time_sec=65.0,
        )
        assert 65.0 < result["efficiency_percent"] <= 85.0
        assert "Good scaling" in result["recommendation"]

    def test_weak_scaling_excellent(self):
        result = calculate_weak_scaling_metrics(
            base_cores=4, base_time_sec=100.0, base_problem_size=64,
            scaled_cores=8, scaled_time_sec=105.0, scaled_problem_size=128,
        )
        assert result["efficiency_percent"] > 90.0
        assert "Excellent weak scaling" in result["recommendation"]

    def test_weak_scaling_good(self):
        result = calculate_weak_scaling_metrics(
            base_cores=4, base_time_sec=100.0, base_problem_size=64,
            scaled_cores=8, scaled_time_sec=120.0, scaled_problem_size=128,
        )
        assert 70.0 < result["efficiency_percent"] <= 90.0
        assert "Good weak scaling" in result["recommendation"]

    def test_weak_scaling_load_imbalance(self):
        result = calculate_weak_scaling_metrics(
            base_cores=4, base_time_sec=100.0, base_problem_size=64,
            scaled_cores=8, scaled_time_sec=110.0, scaled_problem_size=65,
        )
        assert "Load imbalance" in result["recommendation"]

    def test_weak_scaling_invalid_timing(self):
        result = calculate_weak_scaling_metrics(
            base_cores=4, base_time_sec=-1.0, base_problem_size=64,
            scaled_cores=8, scaled_time_sec=50.0, scaled_problem_size=128,
        )
        assert result["speedup"] == 0.0
        assert result["recommendation"] == "Invalid timing data."


class TestGenerateReport:
    def test_report_with_scaling_data(self):
        parsed_scf = {
            "code": "wien2k",
            "converged": True,
            "cpu_time_sec": 20.0,
            "wall_time_sec": 18.0,
            "errors": [],
            "raw_snippet": "",
        }
        report = generate_report(
            parsed_scf=parsed_scf,
            scaling_data={4: 100.0, 8: 55.0},
        )
        assert report.parsing["code"] == "wien2k"
        assert report.scaling is not None
        assert report.scaling["speedup"] == pytest.approx(1.82, rel=0.1)

    def test_report_no_scaling(self):
        parsed_scf = {
            "code": "vasp",
            "converged": False,
            "errors": ["Charge not converged"],
            "raw_snippet": "",
        }
        report = generate_report(parsed_scf=parsed_scf)
        assert report.parsing["code"] == "vasp"
        assert report.scaling is None
        assert any("SCF not converged" in r for r in report.recommendations)

    def test_report_converged(self):
        parsed_scf = {
            "code": "wien2k",
            "converged": True,
            "cpu_time_sec": 95.0,
            "wall_time_sec": 98.0,
            "errors": [],
            "raw_snippet": "",
        }
        report = generate_report(parsed_scf=parsed_scf)
        assert "Excellent CPU utilization" in str(report.recommendations)


class TestAnalysisReport:
    def test_to_dict(self):
        report = AnalysisReport(
            timestamp=1234567890.0,
            code_backend="wien2k",
            recommendations=["Use more cores"],
            warnings=["High memory usage"],
        )
        d = report.to_dict()
        assert d["timestamp"] == 1234567890.0
        assert d["code_backend"] == "wien2k"
        assert d["recommendations"] == ["Use more cores"]
        assert d["warnings"] == ["High memory usage"]

    def test_to_dict_with_none_fields(self):
        report = AnalysisReport(timestamp=0.0, code_backend="qe")
        d = report.to_dict()
        assert d["parsing"] is None
        assert d["scaling"] is None
