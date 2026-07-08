"""
Tests for Wien2kFlags, CalculationType, and ExecutionMode enums.

References:
- Blaha, P. et al. (2020). WIEN2k: An APW+lo program. J. Chem. Phys. 152, 074101.
- WIEN2k Usersguide, Sections 4.1-4.4.
"""

import pytest
from wien2k_gen.types import (
    CalculationType,
    ExecutionMode,
    Wien2kFlags,
    Wien2kVersion,
)


class TestCalculationType:
    """Tests for WIEN2k calculation type detection."""

    def test_default_is_scf(self):
        flags = Wien2kFlags()
        assert flags.get_calculation_type() == CalculationType.SCF

    def test_spin_polarized_detection(self):
        flags = Wien2kFlags(is_spin_polarized=True)
        assert flags.get_calculation_type() == CalculationType.SPIN_POLARIZED

    def test_spin_orbit_detection(self):
        flags = Wien2kFlags(is_soc=True)
        assert flags.get_calculation_type() == CalculationType.SPIN_ORBIT

    def test_spin_polarized_soc_combined(self):
        flags = Wien2kFlags(is_spin_polarized=True, is_soc=True)
        assert flags.get_calculation_type() == CalculationType.SPIN_POLARIZED_SOC

    def test_hybrid_detection(self):
        flags = Wien2kFlags(is_hybrid=True)
        assert flags.get_calculation_type() == CalculationType.HYBRID_FUNC

    def test_lda_u_detection(self):
        flags = Wien2kFlags(is_lda_u=True)
        assert flags.get_calculation_type() == CalculationType.LDA_U

    def test_eece_detection(self):
        flags = Wien2kFlags(is_eece=True)
        assert flags.get_calculation_type() == CalculationType.EECE

    def test_forces_detection(self):
        flags = Wien2kFlags(has_forces=True)
        assert flags.get_calculation_type() == CalculationType.FORCES

    def test_priority_order_spin_over_hybrid(self):
        flags = Wien2kFlags(is_spin_polarized=True, is_hybrid=True)
        assert flags.get_calculation_type() == CalculationType.SPIN_POLARIZED


class TestExecutionCommand:
    """Tests for correct run_lapw/runsp_lapw command generation."""

    def test_default_scf_command(self):
        flags = Wien2kFlags()
        cmd = flags.get_execution_command()
        assert cmd == "run_lapw -p"
        assert "runsp_lapw" not in cmd

    def test_spin_polarized_command(self):
        flags = Wien2kFlags(is_spin_polarized=True)
        cmd = flags.get_execution_command()
        assert cmd.startswith("runsp_lapw")
        assert "-p" in cmd
        assert "run_lapw" not in cmd

    def test_soc_command(self):
        flags = Wien2kFlags(is_soc=True)
        cmd = flags.get_execution_command()
        assert "-so" in cmd
        assert "run_lapw" in cmd

    def test_spin_soc_command(self):
        flags = Wien2kFlags(is_spin_polarized=True, is_soc=True)
        cmd = flags.get_execution_command()
        assert cmd.startswith("runsp_lapw")
        assert "-so" in cmd

    def test_hybrid_command(self):
        flags = Wien2kFlags(is_hybrid=True)
        cmd = flags.get_execution_command()
        assert "-hf" in cmd

    def test_lda_u_command(self):
        flags = Wien2kFlags(is_lda_u=True)
        cmd = flags.get_execution_command()
        assert "-orbc" in cmd

    def test_eece_command(self):
        flags = Wien2kFlags(is_eece=True)
        cmd = flags.get_execution_command()
        assert "-eece" in cmd

    def test_forces_command(self):
        flags = Wien2kFlags(has_forces=True)
        cmd = flags.get_execution_command()
        assert "-fc" in cmd

    def test_combined_flags_command(self):
        flags = Wien2kFlags(is_spin_polarized=True, is_soc=True, is_hybrid=True, is_lda_u=True)
        cmd = flags.get_execution_command()
        assert cmd.startswith("runsp_lapw")
        assert "-so" in cmd
        assert "-orbc" in cmd
        assert "-hf" in cmd


class TestExecutionMode:
    """Tests for ExecutionMode enum including FINE_GRAIN."""

    def test_fine_grain_exists(self):
        assert hasattr(ExecutionMode, "FINE_GRAIN")
        assert ExecutionMode.FINE_GRAIN.value == "fine_grain"

    def test_all_modes_valid(self):
        valid_modes = {"mpi", "hybrid", "kpoint", "serial", "fine_grain"}
        mode_values = {m.value for m in ExecutionMode}
        assert mode_values == valid_modes


class TestWien2kVersion:
    """Tests for WIEN2k version enum."""

    def test_known_versions(self):
        for ver in ["19", "21", "23", "24"]:
            assert Wien2kVersion(ver)

    def test_unknown_fallback(self):
        assert Wien2kVersion.UNKNOWN.value == "unknown"
