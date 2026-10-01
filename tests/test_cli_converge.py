"""CLI converge converts --tolerance from Ry to meV before detection."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

from forge.cli_commands import converge as converge_cmd
from forge.config import AppConfig
from forge.core.constants import RYDBERG_TO_EV
from forge.optimizer.convergence import find_converged_parameters


class _DummyConsole:
    def print(self, *args, **kwargs) -> None:
        return None


def _args(**overrides) -> Namespace:
    values = dict(
        case="case",
        mode="kpoints",
        tolerance=0.001,
        kpoints="2,2,2 4,4,4 6,6,6",
        rkmax="5,6,7",
    )
    values.update(overrides)
    return Namespace(**values)


def test_default_tolerance_ry_converts_to_about_13_6_mev() -> None:
    mev = converge_cmd.tolerance_ry_to_mev(0.001)
    assert mev == RYDBERG_TO_EV * 1000.0 * 0.001
    assert abs(mev - 13.6) < 0.01


def test_handle_passes_mev_not_ry_to_finder(monkeypatch) -> None:
    monkeypatch.setattr(converge_cmd, "get_console", lambda: _DummyConsole())
    captured: list[float] = []
    kpt_data = {"results": [{"parameter": "kpoints", "value": "4x4x4"}]}

    def fake_find(data, tolerance=1.0):
        captured.append(tolerance)
        return {"converged_value": "4x4x4"}

    with patch.multiple(
        "forge.optimizer.convergence",
        run_kpoint_convergence=MagicMock(return_value=kpt_data),
        run_rkmax_convergence=MagicMock(),
        find_converged_parameters=fake_find,
        generate_convergence_report=MagicMock(return_value="ok"),
    ):
        converge_cmd.handle(_args(tolerance=0.001), AppConfig())

    assert len(captured) == 1
    assert captured[0] != 0.001
    assert abs(captured[0] - 13.6) < 0.01


def test_converted_tolerance_picks_earlier_grid_than_buggy_mev() -> None:
    """Old CLI passed 0.001 Ry as 0.001 meV and always took the last grid.

    After conversion, 0.001 Ry ≈ 13.6 meV, so a 0.5 meV step is treated as
    converged at the cheaper 4x4x4 mesh instead of falling through to 6x6x6.
    """
    data = {
        "results": [
            {
                "parameter": "kpoints",
                "value": "2x2x2",
                "total_energy_ry": -100.0,
                "total_energy_ev": -1360.0,
                "delta_energy_mev": 0.0,
                "wall_time_seconds": 10.0,
                "converged": True,
                "n_scf_iterations": 8,
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
                "total_energy_ry": -100.1,
                "total_energy_ev": -1361.0,
                "delta_energy_mev": 0.5,
                "wall_time_seconds": 40.0,
                "converged": True,
                "n_scf_iterations": 10,
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
                "total_energy_ry": -100.11,
                "total_energy_ev": -1361.1,
                "delta_energy_mev": 0.1,
                "wall_time_seconds": 90.0,
                "converged": True,
                "n_scf_iterations": 12,
                "rkmax": 7.0,
                "kpoints": "6x6x6",
                "num_kpoints": 216,
                "success": True,
                "stdout": "",
                "stderr": "",
            },
        ]
    }
    buggy = find_converged_parameters(data, tolerance=0.001)
    fixed = find_converged_parameters(
        data, tolerance=converge_cmd.tolerance_ry_to_mev(0.001)
    )
    assert buggy["converged_value"] == "6x6x6"
    assert fixed["converged_value"] == "4x4x4"
