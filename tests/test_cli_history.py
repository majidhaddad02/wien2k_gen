"""CLI history --similar-to derives nmat/nkpt from the given case.struct."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

from forge.cli_commands import history as history_cmd
from forge.config import AppConfig
from forge.core.case_parser import CaseFileParser

STRUCT = """NaCl rock salt
H   LATTICE,NONEQUIV.ATOMS:  2 225 Fm-3m
MODE OF CALC=RELA unit=bohr
10.0 10.0 10.0 90.0 90.0 90.0
ATOM   1: X=0.00000000 Y=0.00000000 Z=0.00000000
          MULT= 4          ISPLIT= 8
Na1        NPT=  781  R0=.000010000 RMT=    2.50000   Z:  11.0
ATOM   2: X=0.50000000 Y=0.50000000 Z=0.50000000
          MULT= 4          ISPLIT= 8
Cl1        NPT=  781  R0=.000010000 RMT=    2.50000   Z:  17.0
"""

IN1 = """123 TOT
7.0  10  4
0.30 0 0
12.0
"""

KLIST = """          1    0    0    1     1.0
          2    0    1    1     1.0
          3    1    0    1     1.0
          4    1    1    1     1.0
          5    0    0    0     1.0
          6    0    1    0     1.0
          7    1    0    0     1.0
          8    1    1    0     1.0
END
"""

LARGE_STRUCT = """Large cell
H   LATTICE,NONEQUIV.ATOMS:  2 225 Fm-3m
MODE OF CALC=RELA unit=bohr
20.0 20.0 20.0 90.0 90.0 90.0
ATOM   1: X=0.00000000 Y=0.00000000 Z=0.00000000
          MULT= 4          ISPLIT= 8
Na1        NPT=  781  R0=.000010000 RMT=    2.00000   Z:  11.0
ATOM   2: X=0.50000000 Y=0.50000000 Z=0.50000000
          MULT= 4          ISPLIT= 8
Cl1        NPT=  781  R0=.000010000 RMT=    2.00000   Z:  17.0
"""


class _DummyConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, *args, **kwargs) -> None:
        self.messages.append(" ".join(str(a) for a in args))


def _args(similar_to=None, **overrides) -> Namespace:
    values = dict(
        list=False,
        show=None,
        similar_to=similar_to,
        limit=10,
        export=None,
        format="csv",
        backend=None,
    )
    values.update(overrides)
    return Namespace(**values)


def _history_ctx(mock_hist):
    cm = MagicMock()
    cm.__enter__.return_value = mock_hist
    cm.__exit__.return_value = False
    return cm


def test_similar_to_uses_real_nmat_nkpt(tmp_path, monkeypatch) -> None:
    case = tmp_path / "case.struct"
    case.write_text(STRUCT)
    (tmp_path / "case.in1").write_text(IN1)
    (tmp_path / "case.klist").write_text(KLIST)

    struct = CaseFileParser.parse_struct(case)
    expected_nmat = CaseFileParser.estimate_nmat(
        7.0, struct["rmts"], struct["volume_bohr3"]
    )
    expected_nkpt = CaseFileParser.parse_klist(tmp_path / "case.klist")["kpoints"]

    mock_hist = MagicMock()
    mock_hist.get_similar.return_value = []
    mock_hist.query.return_value = []
    monkeypatch.setattr(history_cmd, "get_console", lambda: _DummyConsole())
    with patch(
        "forge.optimizer.history.ExecutionHistory",
        return_value=_history_ctx(mock_hist),
    ):
        history_cmd.handle(_args(similar_to=str(case)), AppConfig())

    mock_hist.get_similar.assert_called_once()
    kwargs = mock_hist.get_similar.call_args.kwargs
    assert kwargs["nmat"] == expected_nmat
    assert kwargs["nkpt"] == expected_nkpt
    assert kwargs["nmat"] != 5000
    assert kwargs["nkpt"] != 4
    mock_hist.query.assert_not_called()


def test_similar_to_without_in1_or_klist_still_computes_nmat(
    tmp_path, monkeypatch
) -> None:
    case = tmp_path / "case.struct"
    case.write_text(STRUCT)

    struct = CaseFileParser.parse_struct(case)
    expected_nmat = CaseFileParser.estimate_nmat(
        7.0, struct["rmts"], struct["volume_bohr3"]
    )

    mock_hist = MagicMock()
    mock_hist.get_similar.return_value = []
    monkeypatch.setattr(history_cmd, "get_console", lambda: _DummyConsole())
    with patch(
        "forge.optimizer.history.ExecutionHistory",
        return_value=_history_ctx(mock_hist),
    ):
        history_cmd.handle(_args(similar_to=str(case)), AppConfig())

    kwargs = mock_hist.get_similar.call_args.kwargs
    assert kwargs["nmat"] == expected_nmat
    assert kwargs["nkpt"] == 1
    assert kwargs["nmat"] != 5000


def test_unparseable_struct_falls_back_to_query(tmp_path, monkeypatch) -> None:
    case = tmp_path / "case.struct"
    case.write_text("")

    console = _DummyConsole()
    mock_hist = MagicMock()
    mock_hist.query.return_value = []
    monkeypatch.setattr(history_cmd, "get_console", lambda: console)
    with patch(
        "forge.optimizer.history.ExecutionHistory",
        return_value=_history_ctx(mock_hist),
    ):
        history_cmd.handle(_args(similar_to=str(case)), AppConfig())

    mock_hist.get_similar.assert_not_called()
    mock_hist.query.assert_called_once_with(limit=10)
    assert any("Could not fully parse" in m for m in console.messages)


def test_non_struct_path_queries_history(tmp_path, monkeypatch) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("not a struct")

    mock_hist = MagicMock()
    mock_hist.query.return_value = []
    monkeypatch.setattr(history_cmd, "get_console", lambda: _DummyConsole())
    with patch(
        "forge.optimizer.history.ExecutionHistory",
        return_value=_history_ctx(mock_hist),
    ):
        history_cmd.handle(_args(similar_to=str(other)), AppConfig())

    mock_hist.get_similar.assert_not_called()
    mock_hist.query.assert_called_once_with(limit=10)


def test_two_structs_yield_different_nmat(tmp_path) -> None:
    small = tmp_path / "small.struct"
    large = tmp_path / "large.struct"
    small.write_text(STRUCT)
    large.write_text(LARGE_STRUCT)
    n_small, _ = history_cmd._nmat_nkpt_from_struct(small)
    n_large, _ = history_cmd._nmat_nkpt_from_struct(large)
    assert n_small != 5000
    assert n_large != 5000
    assert n_large > n_small
