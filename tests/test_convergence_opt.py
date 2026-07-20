"""
Tests for forge.optimizer.convergence — SCF divergence detection,
energy parsing, and convergence reporting.
"""

import contextlib
from unittest.mock import patch

import pytest

from forge.optimizer.convergence import (
    ConvergenceResult,
    _detect_progress_bar,
    _extract_iterations,
    _find_wien2k_commands,
    _modify_incar,
    _modify_klist,
    _parse_mixing_beta,
    _parse_total_energy,
    _run_wien2k_command,
    detect_scf_divergence,
    find_converged_parameters,
    generate_convergence_report,
)

# ---------------------------------------------------------------------------
# Sample SCF output text fixtures (realistic WIEN2k patterns)
# ---------------------------------------------------------------------------


@pytest.fixture
def wien2k_normal_converged() -> str:
    return """\
WIEN2k 23.1
:ENE  : TOTAL ENERGY = -12345.67890
:ENE  : TOTAL ENERGY = -12345.67891
:ENE  : TOTAL ENERGY = -12345.67892
:ENE  : TOTAL ENERGY = -12345.67893
:ENE  : TOTAL ENERGY = -12345.67893
:ENE  : TOTAL ENERGY = -12345.67894
:ENE  : TOTAL ENERGY = -12345.67895
:ENE  : TOTAL ENERGY = -12345.67896
:ENE  : TOTAL ENERGY = -12345.67897
:ENE  : TOTAL ENERGY = -12345.67898
:ENE  : TOTAL ENERGY = -12345.67899
:ENE  : TOTAL ENERGY = -12345.67900
:DIS  : CHARGE CONVERGENCE = 0.00000001
:CPU  : TOTAL CPU TIME FOR SCF IS 100.0
"""


@pytest.fixture
def wien2k_catastrophic() -> str:
    return """\
WIEN2k 23.1
:ENE  : TOTAL ENERGY = -100.0
:ENE  : TOTAL ENERGY = 500.0
:ENE  : TOTAL ENERGY = 5000.0
:ENE  : TOTAL ENERGY = 50000.0
:ENE  : TOTAL ENERGY = 999999.0
QTL-B error detected
"""


@pytest.fixture
def wien2k_monotonic_drift() -> str:
    return "\n".join(
        f":ENE  : TOTAL ENERGY = {-100.0 + i * 0.005}"
        for i in range(20)
    )


@pytest.fixture
def wien2k_charge_sloshing() -> str:
    sign = 1.0
    energies = []
    for i in range(16):
        sign *= -1
        energies.append(-100.0 + sign * 0.01 * (i + 1))
    return "\n".join(
        f":ENE  : TOTAL ENERGY = {e}" for e in energies
    )


@pytest.fixture
def wien2k_stalled() -> str:
    base = -54321.00000001
    lines = [f":ENE  : TOTAL ENERGY = {base}"]
    for _ in range(30):
        lines.append(f":ENE  : TOTAL ENERGY = {base}")
    return "\n".join(lines)


@pytest.fixture
def wien2k_single_cycle() -> str:
    return "WIEN2k 23.1\n:ENE  : TOTAL ENERGY = -500.0\n"


@pytest.fixture
def wien2k_empty() -> str:
    return ""


@pytest.fixture
def wien2k_with_soc() -> str:
    return """\
WIEN2k 23.1  (SOC)
:ENE  : TOTAL ENERGY = -20000.000
:ENE  : TOTAL ENERGY = -20000.001
:ENE  : TOTAL ENERGY = -20000.002
:ENE  : TOTAL ENERGY = -20000.001
:ENE  : TOTAL ENERGY = -20000.002
:ENE  : TOTAL ENERGY = -20000.001
:ENE  : TOTAL ENERGY = -20000.002
:ENE  : TOTAL ENERGY = -20000.001
:ENE  : TOTAL ENERGY = -20000.002
:CPU  : TOTAL CPU TIME FOR SCF IS 200.0
"""


@pytest.fixture
def wien2k_with_hybrid() -> str:
    return """\
WIEN2k 23.1  (hybrid-DFT)
:ENE  : TOTAL ENERGY = -30000.000
:ENE  : TOTAL ENERGY = -30000.001
:ENE  : TOTAL ENERGY = -30000.002
:ENE  : TOTAL ENERGY = -30000.003
:ENE  : TOTAL ENERGY = -30000.004
:ENE  : TOTAL ENERGY = -30000.003
:ENE  : TOTAL ENERGY = -30000.004
:ENE  : TOTAL ENERGY = -30000.003
:ENE  : TOTAL ENERGY = -30000.004
:CPU  : TOTAL CPU TIME FOR SCF IS 500.0
"""


@pytest.fixture
def wien2k_beta_inferred() -> str:
    return """\
WIEN2k 23.1
:MIXING:  PRATT, beta=0.200, cycles=5, reuse=YES
:ENE  : TOTAL ENERGY = -40000.0
:ENE  : TOTAL ENERGY = -40000.001
:ENE  : TOTAL ENERGY = -40000.002
:ENE  : TOTAL ENERGY = -40000.001
:ENE  : TOTAL ENERGY = -40000.002
:ENE  : TOTAL ENERGY = -40000.001
:ENE  : TOTAL ENERGY = -40000.002
:ENE  : TOTAL ENERGY = -40000.001
:ENE  : TOTAL ENERGY = -40000.002
"""


@pytest.fixture
def convergence_results_list_of_dicts() -> list[dict]:
    return [
        {
            "parameter": "kpoints",
            "value": "2x2x2",
            "total_energy_ry": -12345.6789,
            "total_energy_ev": -12345.6789 * 13.6056980659,
            "delta_energy_mev": 0.0,
            "wall_time_seconds": 12.5,
            "converged": True,
            "n_scf_iterations": 10,
            "rkmax": 7.0,
            "kpoints": "2x2x2",
            "num_kpoints": 8,
            "success": True,
            "stdout": "",
            "stderr": "",
        },
        {
            "parameter": "kpoints",
            "value": "4x4x4",
            "total_energy_ry": -12345.6900,
            "total_energy_ev": -12345.6900 * 13.6056980659,
            "delta_energy_mev": 0.5,
            "wall_time_seconds": 45.3,
            "converged": True,
            "n_scf_iterations": 12,
            "rkmax": 7.0,
            "kpoints": "4x4x4",
            "num_kpoints": 64,
            "success": True,
            "stdout": "",
            "stderr": "",
        },
        {
            "parameter": "kpoints",
            "value": "6x6x6",
            "total_energy_ry": -12345.6920,
            "total_energy_ev": -12345.6920 * 13.6056980659,
            "delta_energy_mev": 0.1,
            "wall_time_seconds": 120.0,
            "converged": True,
            "n_scf_iterations": 14,
            "rkmax": 7.0,
            "kpoints": "6x6x6",
            "num_kpoints": 216,
            "success": True,
            "stdout": "",
            "stderr": "",
        },
    ]


@pytest.fixture
def convergence_results_as_objects(
    convergence_results_list_of_dicts,
) -> list[ConvergenceResult]:
    return [
        ConvergenceResult.from_dict(d) for d in convergence_results_list_of_dicts
    ]


# ---------------------------------------------------------------------------
# ConvergenceResult dataclass
# ---------------------------------------------------------------------------


class TestConvergenceResult:
    def test_creation_defaults(self):
        cr = ConvergenceResult(
            parameter="rkmax",
            value=7.0,
            total_energy_ry=-100.0,
            total_energy_ev=-1360.57,
            delta_energy_mev=0.5,
            wall_time_seconds=30.0,
            converged=True,
            n_scf_iterations=15,
            rkmax=7.0,
            kpoints="4x4x4",
            num_kpoints=64,
            success=True,
            stdout="lapw0 output",
            stderr="",
        )
        assert cr.parameter == "rkmax"
        assert cr.value == 7.0
        assert cr.total_energy_ry == -100.0
        assert cr.converged is True
        assert cr.n_scf_iterations == 15
        assert cr.num_kpoints == 64

    def test_to_dict(self):
        cr = ConvergenceResult(
            parameter="kpoints",
            value="2x2x2",
            total_energy_ry=-50.0,
            total_energy_ev=-680.285,
            delta_energy_mev=1.0,
            wall_time_seconds=10.0,
            converged=False,
            n_scf_iterations=5,
            rkmax=6.0,
            kpoints="2x2x2",
            num_kpoints=8,
            success=False,
            stdout="",
            stderr="error",
        )
        d = cr.to_dict()
        assert isinstance(d, dict)
        assert d["parameter"] == "kpoints"
        assert d["value"] == "2x2x2"
        assert d["success"] is False
        assert d["stderr"] == "error"

    def test_from_dict(self):
        data = {
            "parameter": "rkmax",
            "value": 8.0,
            "total_energy_ry": -200.0,
            "total_energy_ev": -2721.14,
            "delta_energy_mev": 0.0,
            "wall_time_seconds": 60.0,
            "converged": True,
            "n_scf_iterations": 20,
            "rkmax": 8.0,
            "kpoints": "6x6x6",
            "num_kpoints": 216,
            "success": True,
            "stdout": "done",
            "stderr": "",
        }
        cr = ConvergenceResult.from_dict(data)
        assert cr.parameter == "rkmax"
        assert cr.value == 8.0
        assert cr.total_energy_ev == -2721.14

    def test_roundtrip_to_from_dict(self):
        original = ConvergenceResult(
            parameter="kpoints",
            value="3x3x3",
            total_energy_ry=-300.0,
            total_energy_ev=-4081.71,
            delta_energy_mev=2.5,
            wall_time_seconds=90.0,
            converged=True,
            n_scf_iterations=12,
            rkmax=7.5,
            kpoints="3x3x3",
            num_kpoints=27,
            success=True,
            stdout="...",
            stderr="",
        )
        restored = ConvergenceResult.from_dict(original.to_dict())
        assert restored.parameter == original.parameter
        assert restored.value == original.value
        assert restored.total_energy_ry == original.total_energy_ry
        assert restored.delta_energy_mev == original.delta_energy_mev


# ---------------------------------------------------------------------------
# _parse_total_energy
# ---------------------------------------------------------------------------


class TestParseTotalEnergy:
    def test_extracts_last_ene(self):
        lines = [
            ":ENE  : TOTAL ENERGY = -12345.67890",
            "lapw0 : cpu time : 10.5",
            ":ENE  : TOTAL ENERGY = -12345.99999",
        ]
        result = _parse_total_energy(lines)
        assert result == pytest.approx(-12345.99999)

    def test_no_ene_tag_returns_zero(self):
        lines = ["lapw0 : cpu time : 10.5", "lapw1 : cpu time : 45.2"]
        result = _parse_total_energy(lines)
        assert result == 0.0

    def test_empty_lines(self):
        assert _parse_total_energy([]) == 0.0

    def test_garbage_after_ene(self):
        lines = [":ENE  : TOTAL ENERGY = xyz_abc  -500.0"]
        result = _parse_total_energy(lines)
        assert result == -500.0

    def test_multiple_numbers_on_ene_line(self):
        lines = [":ENE  : TOTAL ENERGY = -300.0 0.0 100.0"]
        result = _parse_total_energy(lines)
        assert result == -300.0

    def test_ene_without_parseable_tokens(self):
        lines = [":ENE  : TOTAL ENERGY =?"]
        result = _parse_total_energy(lines)
        assert result == 0.0

    def test_ene_empty_after_tag(self):
        lines = [":ENE  :"]
        result = _parse_total_energy(lines)
        assert result == 0.0


# ---------------------------------------------------------------------------
# _extract_iterations
# ---------------------------------------------------------------------------


class TestExtractIterations:
    def test_counts_iter_tags(self):
        lines = [
            ":ITER 1",
            ":ITER 2",
            ":ITER 3",
            ":ITER 4",
        ]
        assert _extract_iterations(lines) == 4

    def test_no_iter_tags(self):
        lines = [":ENE  : TOTAL ENERGY = -500.0", "lapw0 : cpu time : 10.5"]
        assert _extract_iterations(lines) == 0

    def test_empty_lines(self):
        assert _extract_iterations([]) == 0


# ---------------------------------------------------------------------------
# _parse_mixing_beta
# ---------------------------------------------------------------------------


class TestParseMixingBeta:
    def test_mixing_keyword(self):
        content = ":MIXING:  MSR1, beta=0.150, cycles=5, reuse=YES\n:ENE  : TOTAL ENERGY = -100.0"
        assert _parse_mixing_beta(content) == 0.15

    def test_pratt_keyword(self):
        content = ":MIXING:  PRATT, beta=0.080, cycles=3, reuse=NO\n:ENE  : TOTAL ENERGY = -100.0"
        assert _parse_mixing_beta(content) == 0.08

    def test_beta_keyword(self):
        content = ":MIXING:  MSEC, beta=0.250, cycles=5\n:ENE  : TOTAL ENERGY = -100.0"
        assert _parse_mixing_beta(content) == 0.25

    def test_msec_keyword(self):
        content = ":MIXING:  MSEC, beta=0.100\n:ENE  : TOTAL ENERGY = -100.0"
        assert _parse_mixing_beta(content) == 0.10

    def test_no_mixing_info(self):
        content = "lapw0 : cpu time : 10.5\n:ENE  : TOTAL ENERGY = -100.0"
        assert _parse_mixing_beta(content) == 0.0

    def test_case_insensitive(self):
        content = ":mixing:  msr1a, BETA=0.120\n:ENE  : TOTAL ENERGY = -100.0"
        assert _parse_mixing_beta(content) == 0.12


# ---------------------------------------------------------------------------
# detect_scf_divergence
# ---------------------------------------------------------------------------


class TestDetectScfDivergence:
    # -- Normal convergence --

    def test_normal_convergence_from_text(
        self, wien2k_normal_converged,
    ):
        result = detect_scf_divergence(wien2k_normal_converged)
        assert result["divergent"] is False
        assert result["divergence_type"] == "none"
        assert result["severity"] == 0.0

    def test_normal_convergence_from_values(self):
        energies = [-100.0, -100.001, -100.002, -100.003, -100.004,
                    -100.005, -100.006]
        result = detect_scf_divergence("", energy_values=energies)
        assert result["divergent"] is False
        assert result["divergence_type"] == "none"

    # -- Few cycles (return defaults) --

    def test_few_cycles_returns_no_divergence(self):
        energies = [-100.0, -100.1]
        result = detect_scf_divergence("", energy_values=energies)
        assert result["divergent"] is False
        assert result["divergence_type"] == "none"
        assert result["severity"] == 0.0
        assert result["recommended_action"] == ""

    def test_empty_content(self, wien2k_empty):
        result = detect_scf_divergence(wien2k_empty)
        assert result["divergent"] is False
        assert result["divergence_type"] == "none"

    def test_single_cycle(self, wien2k_single_cycle):
        result = detect_scf_divergence(wien2k_single_cycle)
        assert result["divergent"] is False

    # -- Catastrophic divergence --

    def test_catastrophic_divergence(self, wien2k_catastrophic):
        result = detect_scf_divergence(wien2k_catastrophic)
        assert result["divergent"] is True
        assert result["divergence_type"] == "catastrophic"
        assert result["severity"] == 1.0
        assert "RMT" in result["recommended_action"]
        assert result["auto_mixing_params"]["beta"] == 0.02
        assert result["auto_mixing_params"]["pratt_cycles"] == 10

    def test_catastrophic_from_values(self):
        energies = [-10.0, -5.0, 200000.0, 300000.0, 500000.0]
        result = detect_scf_divergence("", energy_values=energies)
        assert result["divergent"] is True
        assert result["divergence_type"] == "catastrophic"

    # -- Monotonic drift --

    def test_monotonic_drift(self, wien2k_monotonic_drift):
        result = detect_scf_divergence(wien2k_monotonic_drift)
        assert result["divergent"] is True
        assert result["divergence_type"] == "monotonic_drift"
        assert result["severity"] > 0.0
        assert "drifting" in result["recommended_action"].lower()
        assert result["auto_mixing_params"]["beta"] == 0.03
        assert result["auto_mixing_params"]["pratt_cycles"] == 5
        assert result["auto_mixing_params"]["msr1a"] is True

    def test_monotonic_drift_from_values(self):
        energies = [-100.0 + i * 0.01 for i in range(15)]
        result = detect_scf_divergence("", energy_values=energies)
        assert result["divergent"] is True
        assert result["divergence_type"] == "monotonic_drift"

    # -- Charge sloshing --

    def test_charge_sloshing(self, wien2k_charge_sloshing):
        result = detect_scf_divergence(wien2k_charge_sloshing)
        assert result["divergent"] is True
        assert result["divergence_type"] == "charge_sloshing"
        assert "sloshing" in result["recommended_action"].lower()
        assert result["auto_mixing_params"]["beta"] > 0.0
        assert result["auto_mixing_params"]["pratt_cycles"] == 3

    def test_charge_sloshing_with_beta_inference(
        self, wien2k_beta_inferred,
    ):
        result = detect_scf_divergence(wien2k_beta_inferred)
        assert result["divergent"] is True
        assert result["divergence_type"] == "charge_sloshing"
        assert result["auto_mixing_params"]["beta"] == 0.10

    def test_charge_sloshing_from_values(self):
        energies = [
            -100.0, -99.99, -100.01, -99.98, -100.02,
            -99.97, -100.03, -99.96, -100.04, -99.95,
        ]
        result = detect_scf_divergence("", energy_values=energies)
        assert result["divergent"] is True
        assert result["divergence_type"] == "charge_sloshing"

    # -- Stalled convergence --

    def test_stalled(self, wien2k_stalled):
        result = detect_scf_divergence(wien2k_stalled)
        assert result["divergent"] is True
        assert result["divergence_type"] == "stalled"
        assert result["severity"] == 0.5
        assert "stalled" in result["recommended_action"].lower()
        assert result["auto_mixing_params"]["beta"] == 0.15

    def test_stalled_from_values(self):
        energies = [-54321.0] + [-54321.0] * 30
        result = detect_scf_divergence("", energy_values=energies)
        assert result["divergent"] is True
        assert result["divergence_type"] == "stalled"

    # -- 100+ cycles --

    def test_100plus_cycles(self):
        energies = []
        for i in range(120):
            if i % 2 == 0:
                energies.append(-54321.0 + i * 0.0001)
            else:
                energies.append(-54321.0 - i * 0.0001)
        result = detect_scf_divergence("", energy_values=energies)
        assert result["divergent"] is True
        assert result["divergence_type"] == "charge_sloshing"

    # -- SOC flagged --

    def test_soc_output_normal(self, wien2k_with_soc):
        result = detect_scf_divergence(wien2k_with_soc)
        assert result["divergence_type"] == "charge_sloshing"

    # -- Hybrid flagged --

    def test_hybrid_output(self, wien2k_with_hybrid):
        result = detect_scf_divergence(wien2k_with_hybrid)
        assert result["divergence_type"] == "charge_sloshing"

    # -- Result structure --

    def test_result_always_has_required_keys(self):
        result = detect_scf_divergence("", energy_values=[-100.0, -100.1])
        for key in (
            "divergent",
            "divergence_type",
            "severity",
            "recommended_action",
            "auto_mixing_params",
        ):
            assert key in result
        assert "beta" in result["auto_mixing_params"]
        assert "pratt_cycles" in result["auto_mixing_params"]
        assert "msr1a" in result["auto_mixing_params"]

    # -- catastrophe is detected before monotonic drift --

    def test_catastrophic_takes_priority_over_drift(self):
        energies = [-500.0 + i * 0.01 for i in range(10)]
        energies.append(1e8)
        result = detect_scf_divergence("", energy_values=energies)
        assert result["divergent"] is True
        assert result["divergence_type"] == "catastrophic"


# ---------------------------------------------------------------------------
# find_converged_parameters
# ---------------------------------------------------------------------------


class TestFindConvergedParameters:
    def test_finds_first_below_tolerance(
        self, convergence_results_list_of_dicts,
    ):
        data = {"results": convergence_results_list_of_dicts}
        result = find_converged_parameters(data, tolerance=1.0)
        assert result["parameter"] == "kpoints"
        assert result["converged_value"] == "4x4x4"
        assert result["delta_mev"] == 0.5

    def test_falls_back_to_last_when_none_below_tolerance(
        self, convergence_results_list_of_dicts,
    ):
        data = {"results": convergence_results_list_of_dicts}
        result = find_converged_parameters(data, tolerance=0.001)
        assert result["converged_value"] == "6x6x6"

    def test_empty_results(self):
        result = find_converged_parameters({"results": []})
        assert result["parameter"] == "unknown"
        assert result["converged_value"] is None
        assert result["energy_ev"] == 0.0

    def test_missing_results_key(self):
        result = find_converged_parameters({})
        assert result["parameter"] == "unknown"

    def test_delta_zero_not_treated_as_converged(self):
        data = {
            "results": [
                {
                    "parameter": "rkmax",
                    "value": 5.0,
                    "total_energy_ry": -73.5,
                    "total_energy_ev": -1000.0,
                    "delta_energy_mev": 0.0,
                    "wall_time_seconds": 10.0,
                    "converged": True,
                    "n_scf_iterations": 5,
                    "rkmax": 5.0,
                    "kpoints": "2x2x2",
                    "num_kpoints": 8,
                    "success": True,
                    "stdout": "",
                    "stderr": "",
                },
                {
                    "parameter": "rkmax",
                    "value": 6.0,
                    "total_energy_ry": -73.54,
                    "total_energy_ev": -1000.5,
                    "delta_energy_mev": 0.5,
                    "wall_time_seconds": 12.0,
                    "converged": True,
                    "n_scf_iterations": 6,
                    "rkmax": 6.0,
                    "kpoints": "2x2x2",
                    "num_kpoints": 8,
                    "success": True,
                    "stdout": "",
                    "stderr": "",
                },
            ]
        }
        result = find_converged_parameters(data, tolerance=1.0)
        assert result["converged_value"] == 6.0

    def test_works_with_convergence_result_objects(
        self, convergence_results_as_objects,
    ):
        data = {"results": convergence_results_as_objects}
        result = find_converged_parameters(data, tolerance=1.0)
        assert result["parameter"] == "kpoints"
        assert result["converged_value"] == "4x4x4"


# ---------------------------------------------------------------------------
# generate_convergence_report
# ---------------------------------------------------------------------------


class TestGenerateConvergenceReport:
    def test_generates_report_with_results(
        self, convergence_results_list_of_dicts,
    ):
        data = {"results": convergence_results_list_of_dicts}
        report = generate_convergence_report(data)
        assert "Convergence Study Report" in report
        assert "kpoints" in report
        assert "2x2x2" in report
        assert "4x4x4" in report
        assert "6x6x6" in report
        assert "Convergence Summary" in report

    def test_empty_results(self):
        report = generate_convergence_report({"results": []})
        assert "No convergence data available" in report

    def test_works_with_convergence_result_objects(
        self, convergence_results_as_objects,
    ):
        data = {"results": convergence_results_as_objects}
        report = generate_convergence_report(data)
        assert "Convergence Study Report" in report
        assert "2x2x2" in report

    def test_report_contains_header_footer(self):
        data = {
            "results": [
                {
                    "parameter": "rkmax",
                    "value": 7.0,
                    "total_energy_ry": -36.75,
                    "total_energy_ev": -500.0,
                    "delta_energy_mev": 0.1,
                    "wall_time_seconds": 30.0,
                    "n_scf_iterations": 10,
                    "success": True,
                    "converged": True,
                    "rkmax": 7.0,
                    "kpoints": "4x4x4",
                    "num_kpoints": 64,
                    "stdout": "",
                    "stderr": "",
                }
            ]
        }
        report = generate_convergence_report(data)
        assert report.startswith("=" * 72)
        assert report.endswith("=" * 72)
        assert "OK" in report

    def test_failure_status_in_report(self):
        data = {
            "results": [
                {
                    "parameter": "rkmax",
                    "value": 9.0,
                    "total_energy_ry": -44.1,
                    "total_energy_ev": -600.0,
                    "delta_energy_mev": 5.0,
                    "wall_time_seconds": 90.0,
                    "n_scf_iterations": 40,
                    "success": False,
                    "converged": False,
                    "rkmax": 9.0,
                    "kpoints": "4x4x4",
                    "num_kpoints": 64,
                    "stdout": "",
                    "stderr": "",
                }
            ]
        }
        report = generate_convergence_report(data)
        assert "FAIL" in report


# ---------------------------------------------------------------------------
# Edge / branch coverage for detect_scf_divergence
# ---------------------------------------------------------------------------


class TestDetectScfDivergenceBranches:
    def test_20_cycles_not_stalled(self):
        """20+ cycles with non-trivial deltas => not stalled."""
        energies = [-100.0]
        for i in range(30):
            energies.append(energies[-1] - 0.000001 * (i + 1))
        result = detect_scf_divergence("", energy_values=energies)
        assert result["divergent"] is False
        assert result["divergence_type"] == "none"

    def test_sloshing_no_ene_in_text_empty_energy_values(self):
        """Sloshing with no :ENE in text => _parse_mixing_beta returns 0 => beta=0.05."""
        energies = []
        base = -100.0
        for i in range(16):
            energies.append(base + (0.01 if i % 2 == 0 else -0.01) * (i + 1))
        result = detect_scf_divergence("no ene tags\n" * 16, energy_values=energies)
        assert result["divergent"] is True
        assert result["divergence_type"] == "charge_sloshing"
        assert result["auto_mixing_params"]["beta"] == 0.05


# ---------------------------------------------------------------------------
# _run_wien2k_command
# ---------------------------------------------------------------------------


class TestRunWien2kCommand:
    def test_successful_run(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ":ENE  : TOTAL ENERGY = -100.0"
            mock_run.return_value.stderr = ""
            rc, stdout, _ = _run_wien2k_command(["run_lapw"], tmp_path)
            assert rc == 0
            assert ":ENE" in stdout

    def test_timeout(self, tmp_path):
        with patch("subprocess.run", side_effect=subprocess_TimeoutExpired):
            rc, _, stderr = _run_wien2k_command(["run_lapw"], tmp_path)
            assert rc == -1
            assert "Timeout" in stderr

    def test_file_not_found(self, tmp_path):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            rc, _, stderr = _run_wien2k_command(["nonexistent"], tmp_path)
            assert rc == -1
            assert "not found" in stderr

    def test_string_command_converted_to_list(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "ok"
            mock_run.return_value.stderr = ""
            _run_wien2k_command("run_lapw -p", tmp_path)
            # Verify list was passed
            assert isinstance(mock_run.call_args[0][0], list)


subprocess_TimeoutExpired = __import__("subprocess").TimeoutExpired


# ---------------------------------------------------------------------------
# _detect_progress_bar
# ---------------------------------------------------------------------------


class TestDetectProgressBar:
    def test_returns_type_string(self):
        result = _detect_progress_bar()
        assert result[0] in ("rich", "tqdm", "none")

    def test_none_when_both_unavailable(self):
        with patch.dict("sys.modules", {"rich.progress": None, "tqdm": None}):
            import importlib
            with contextlib.suppress(ImportError, ModuleNotFoundError):
                importlib.reload(
                    __import__("forge.optimizer.convergence", fromlist=["_detect_progress_bar"])
                )
        result = _detect_progress_bar()
        assert result[0] in ("rich", "tqdm", "none")


# ---------------------------------------------------------------------------
# _modify_klist
# ---------------------------------------------------------------------------


class TestModifyKlist:
    def test_writes_klist_file(self, tmp_path):
        klist = tmp_path / "case.klist"
        _modify_klist(klist, 8, 8, 8)
        content = klist.read_text()
        assert "8" in content
        assert "1.5" in content
        assert "END" in content

    def test_different_grid_sizes(self, tmp_path):
        klist = tmp_path / "case.klist"
        _modify_klist(klist, 12, 12, 1)
        content = klist.read_text()
        assert "12" in content
        assert "1" in content


# ---------------------------------------------------------------------------
# _modify_incar
# ---------------------------------------------------------------------------


class TestModifyIncar:
    def test_rkmax_update(self, tmp_path):
        in1 = tmp_path / "case.in1"
        in1.write_text("RKMAX  7.0\nOTHER  value\n")
        _modify_incar(in1, {"RKMAX": "8.5"})
        content = in1.read_text()
        assert "8.5" in content
        assert "# RKMAX" in content

    def test_other_keywords_preserved(self, tmp_path):
        in1 = tmp_path / "case.in1"
        original = "RKMAX  7.0\nOTHER  value\nEND\n"
        in1.write_text(original)
        _modify_incar(in1, {"RKMAX": "6.0"})
        content = in1.read_text()
        assert "OTHER" in content
        assert "END" in content

    def test_nonexistent_file_returns_empty(self, tmp_path):
        in1 = tmp_path / "nonexistent.in1"
        result = _modify_incar(in1, {"RKMAX": "5.0"})
        assert result == []

    def test_empty_file(self, tmp_path):
        in1 = tmp_path / "case.in1"
        in1.write_text("")
        result = _modify_incar(in1, {"RKMAX": "5.0"})
        assert result == []

    def test_preserves_blank_lines(self, tmp_path):
        in1 = tmp_path / "case.in1"
        in1.write_text("RKMAX  7.0\n\nOTHER  value\n")
        _modify_incar(in1, {"RKMAX": "9.0"})
        content = in1.read_text()
        lines = content.splitlines()
        assert lines[1] == ""

    def test_non_rkmax_key_matching_line(self, tmp_path):
        """Line matches a key that is not RKMAX — branch at line 220."""
        in1 = tmp_path / "case.in1"
        in1.write_text("KPOINT  4\nRKMAX  7.0\n")
        _modify_incar(in1, {"KPOINT": "8"})
        content = in1.read_text()
        assert "KPOINT  4" in content
        assert "RKMAX  7.0" in content


# ---------------------------------------------------------------------------
# _find_wien2k_commands
# ---------------------------------------------------------------------------


class TestFindWien2kCommands:
    def test_absolute_paths_preserved(self):
        cmds = {"run_lapw": "/usr/local/bin/run_lapw"}
        resolved = _find_wien2k_commands(cmds)
        assert resolved["run_lapw"] == "/usr/local/bin/run_lapw"

    def test_relative_path_uses_shutil_which(self):
        with patch("shutil.which", return_value="/usr/bin/run_lapw"):
            cmds = {"run_lapw": "run_lapw"}
            resolved = _find_wien2k_commands(cmds)
            assert resolved["run_lapw"] == "/usr/bin/run_lapw"

    def test_relative_path_not_found_falls_back(self):
        with patch("shutil.which", return_value=None):
            cmds = {"run_lapw": "run_lapw"}
            resolved = _find_wien2k_commands(cmds)
            assert resolved["run_lapw"] == "run_lapw"

    def test_multiple_commands(self):
        cmds = {
            "init_lapw": "/opt/wien2k/init_lapw",
            "run_lapw": "run_lapw",
        }
        with patch("shutil.which", return_value="/usr/bin/run_lapw"):
            resolved = _find_wien2k_commands(cmds)
            assert resolved["init_lapw"] == "/opt/wien2k/init_lapw"
            assert resolved["run_lapw"] == "/usr/bin/run_lapw"
