"""Integration tests for WIEN2k flag detection and execution command generation."""

import tempfile
from pathlib import Path

from wien2k_gen.types import CalculationType, Wien2kFlags, Wien2kVersion


def _touch(path: Path) -> Path:
    path.write_text("", encoding="utf-8")
    return path


def test_default_flags_are_false() -> None:
    f = Wien2kFlags()
    assert f.is_spin_polarized is False
    assert f.is_soc is False
    assert f.is_lda_u is False
    assert f.is_hybrid is False
    assert f.is_eece is False
    assert f.has_forces is False


def test_default_calculation_type_scf() -> None:
    f = Wien2kFlags()
    assert f.get_calculation_type() == CalculationType.SCF


def test_default_exec_command() -> None:
    f = Wien2kFlags()
    assert f.get_execution_command() == "run_lapw -p"


def test_spin_polarized_calculation_type() -> None:
    f = Wien2kFlags(is_spin_polarized=True)
    assert f.get_calculation_type() == CalculationType.SPIN_POLARIZED


def test_spin_polarized_exec_command() -> None:
    f = Wien2kFlags(is_spin_polarized=True)
    assert f.get_execution_command() == "runsp_lapw -p"


def test_soc_exec_command() -> None:
    f = Wien2kFlags(is_soc=True)
    assert f.get_execution_command() == "run_lapw -p -so"


def test_spin_polarized_soc_exec_command() -> None:
    f = Wien2kFlags(is_spin_polarized=True, is_soc=True)
    assert f.get_calculation_type() == CalculationType.SPIN_POLARIZED_SOC
    assert f.get_execution_command() == "runsp_lapw -p -so"


def test_lda_u_exec_command() -> None:
    f = Wien2kFlags(is_lda_u=True)
    assert f.get_calculation_type() == CalculationType.LDA_U
    assert f.get_execution_command() == "run_lapw -p -orbc"


def test_hybrid_exec_command() -> None:
    f = Wien2kFlags(is_hybrid=True)
    assert f.get_calculation_type() == CalculationType.HYBRID_FUNC
    assert f.get_execution_command() == "run_lapw -p -hf"


def test_eece_exec_command() -> None:
    f = Wien2kFlags(is_eece=True)
    assert f.get_calculation_type() == CalculationType.EECE
    assert f.get_execution_command() == "run_lapw -p -eece"


def test_forces_exec_command() -> None:
    f = Wien2kFlags(has_forces=True)
    assert f.get_calculation_type() == CalculationType.FORCES
    assert f.get_execution_command() == "run_lapw -p -fc"


def test_multiple_flags_combined() -> None:
    f = Wien2kFlags(is_spin_polarized=True, is_soc=True, is_lda_u=True)
    assert f.get_calculation_type() == CalculationType.SPIN_POLARIZED_SOC
    cmd = f.get_execution_command()
    assert "-so" in cmd
    assert "-orbc" in cmd


def test_spin_overrides_everything() -> None:
    f = Wien2kFlags(is_spin_polarized=True, is_lda_u=True, is_hybrid=True, is_eece=True)
    assert f.get_calculation_type() == CalculationType.SPIN_POLARIZED
    assert "runsp_lapw" in f.get_execution_command()
    assert "-orbc" in f.get_execution_command()
    assert "-hf" in f.get_execution_command()
    assert "-eece" in f.get_execution_command()


def test_wien2k_version_string() -> None:
    f = Wien2kFlags(wien2k_version="23.1")
    assert f.wien2k_version == "23.1"


def test_calculation_types_coverage() -> None:
    """Verify all 8 calculation types are defined."""
    from wien2k_gen.types import CalculationType as CT
    expected = {
        CT.SCF, CT.SPIN_POLARIZED, CT.SPIN_ORBIT, CT.SPIN_POLARIZED_SOC,
        CT.LDA_U, CT.HYBRID_FUNC, CT.FORCES, CT.EECE,
    }
    for ct in CalculationType:
        assert ct in expected
    assert len(list(CalculationType)) == 8
