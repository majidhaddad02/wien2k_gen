"""Tests for forge diagnose — SCF parser flags and healthy-status gating."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from forge.cli_commands import diagnose as diagnose_cmd
from forge.core.case_parser import parse_scf_output

FIXTURES = Path(__file__).parent / "fixtures"
HEALTHY_MSG = "No critical issues detected. SCF appears healthy."
CRASH_TITLE = "LAPWx Crash Detected"
QTLB_TITLE = "QTL-B Error"


class CapturingConsole:
    def __init__(self) -> None:
        self.items: list[object] = []

    def print(self, *args, **kwargs) -> None:
        self.items.extend(args)

    def rendered(self) -> str:
        parts: list[str] = []
        for item in self.items:
            title = getattr(item, "title", None)
            renderable = getattr(item, "renderable", None)
            if title is not None:
                parts.append(str(title))
            if renderable is not None:
                parts.append(str(renderable))
            parts.append(str(item))
        return "\n".join(parts)


def _args(log: Path) -> Namespace:
    return Namespace(case=None, log=str(log))


def _stub_optional_analyzers(monkeypatch) -> None:
    monkeypatch.setattr(
        "forge.optimizer.monitor.diagnose_charge_sloshing_root_cause",
        lambda *a, **k: {"root_cause": "none", "confidence": 0.0, "actions": []},
    )
    monkeypatch.setattr(
        "forge.optimizer.convergence.detect_scf_divergence",
        lambda *a, **k: {
            "divergent": False,
            "divergence_type": "",
            "severity": 0.0,
            "recommended_action": "",
            "auto_mixing_params": {"beta": 0, "pratt_cycles": 0, "msr1a": False},
        },
    )


def _run(log: Path, monkeypatch, console: CapturingConsole | None = None) -> tuple[dict, CapturingConsole]:
    cap = console or CapturingConsole()
    _stub_optional_analyzers(monkeypatch)
    monkeypatch.setattr(diagnose_cmd, "get_console", lambda: cap)
    result = diagnose_cmd.handle(_args(log), None)
    return result, cap


def test_converged_clean_run_prints_healthy(tmp_path, monkeypatch):
    result, cap = _run(FIXTURES / "scf_converged.scf", monkeypatch)
    text = cap.rendered()
    assert result["converged"] is True
    assert result["has_qtlb"] is False
    assert result["has_crash"] is False
    assert result["has_not_conv"] is False
    assert HEALTHY_MSG in text
    # Bug 2 regression: old inverted `not any(..., converged, ...)` hid this
    # message whenever converged was True. It must print for a clean success.
    assert "healthy" in text.lower()


def test_real_lapw1_crash_sets_has_crash_and_shows_panel(tmp_path, monkeypatch):
    scf = tmp_path / "crash.scf"
    scf.write_text(
        ":ENE  : TOTAL ENERGY    =      -1000.00000000\n"
        ":DIS  : CHARGE CONVERGENCE = 0.00000005\n"
        "ERROR: lapw1 crashed in MPI communication\n"
    )
    parsed = parse_scf_output(scf.read_text())
    assert any(e.startswith("LAPWx crashed") for e in parsed["errors"])
    result, cap = _run(scf, monkeypatch)
    assert result["has_crash"] is True
    assert CRASH_TITLE in cap.rendered()
    # Old literal `"lapw crashed"` (no stage digit) would miss this real message.
    assert "lapw crashed" not in scf.read_text().lower()


def test_synthetic_lapw_crashed_without_digit_is_not_real_wien2k():
    """Old diagnose.py scanned for the literal 'lapw crashed', which real WIEN2k never emits."""
    synthetic = "synthetic log containing lapw crashed with no stage digit\n"
    assert "lapw crashed" in synthetic.lower()
    assert "lapw1 crashed" not in synthetic.lower()
    assert "lapw0 crashed" not in synthetic.lower()
    assert "lapw2 crashed" not in synthetic.lower()


def test_qtlb_derived_from_parsed_errors_and_remediation_intact(tmp_path, monkeypatch):
    scf = tmp_path / "qtlb.scf"
    scf.write_text(
        ":ENE  : TOTAL ENERGY    =      -12345.67890123\n"
        ":DIS  : CHARGE CONVERGENCE = 0.00050000\n"
        "QTL-B error in atom 3\n"
        "rkmax too large for this basis\n"
        "sphere overlap detected\n"
        "linearization energy missing in case.in1\n"
    )
    content = scf.read_text()
    parsed = parse_scf_output(content)
    assert any(e.startswith("QTL-B") for e in parsed["errors"])
    result, cap = _run(scf, monkeypatch)
    text = cap.rendered()
    assert result["has_qtlb"] is True
    assert result["has_qtlb"] == any(e.startswith("QTL-B") for e in parsed["errors"])
    assert QTLB_TITLE in text
    assert "Reduce RKMAX" in text
    assert "Reduce RMT" in text or "sphere overlap" in text.lower()
    assert "linearization energies" in text


def test_not_converged_does_not_print_healthy(monkeypatch):
    result, cap = _run(FIXTURES / "scf_not_converged.scf", monkeypatch)
    text = cap.rendered()
    assert result["converged"] is False
    assert result["has_not_conv"] is True
    assert HEALTHY_MSG not in text
    # Bug 2 old inverted condition would have printed "healthy" here:
    # not any([False, False, True, False, False]) is False because has_not_conv
    # is True for this fixture — wait, this fixture HAS "not converged" so the
    # old path also would suppress healthy. The inverted-condition regression
    # is a non-converged file WITHOUT explicit failure/error flags.
    # Covered separately below for the exact old inverted case.


def test_unconverged_without_other_flags_is_not_healthy(tmp_path, monkeypatch):
    """Old inverted condition printed healthy when converged=False and no error flags."""
    scf = tmp_path / "slow.scf"
    scf.write_text(
        ":ENE  : TOTAL ENERGY    =      -10000.00000000\n"
        ":DIS  : CHARGE CONVERGENCE = 0.00100000\n"
        ":ENE  : TOTAL ENERGY    =       -9999.50000000\n"
        ":DIS  : CHARGE CONVERGENCE = 0.00090000\n"
    )
    parsed = parse_scf_output(scf.read_text())
    assert parsed["converged"] is False
    assert parsed["explicit_failure"] is False
    assert parsed["errors"] == []
    result, cap = _run(scf, monkeypatch)
    assert result["converged"] is False
    assert result["has_qtlb"] is False
    assert result["has_crash"] is False
    assert result["has_not_conv"] is False
    assert HEALTHY_MSG not in cap.rendered()
