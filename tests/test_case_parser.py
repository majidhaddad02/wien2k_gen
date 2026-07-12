"""Tests for wien2k_gen.core.case_parser — WIEN2k input file parsing."""

import tempfile
from pathlib import Path

import pytest

from wien2k_gen.core.case_parser import (
    CaseFileParser,
    LDAUData,
    Vector,
    parse_case_directory,
    try_float,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


# ============================================================
# Vector
# ============================================================


def test_vector_dot() -> None:
    a = Vector(1, 0, 0)
    b = Vector(0, 1, 0)
    assert a.dot(b) == 0.0
    assert a.dot(Vector(2, 0, 0)) == 2.0


def test_vector_cross() -> None:
    a = Vector(1, 0, 0)
    b = Vector(0, 1, 0)
    c = a.cross(b)
    assert c.x == 0
    assert c.y == 0
    assert c.z == 1


def test_try_float() -> None:
    assert try_float("1.23") == 1.23
    assert try_float(" 4 ") == 4.0
    assert try_float("abc") is None
    assert try_float("") is None


# ============================================================
# case.in1 parsing
# ============================================================


def test_parse_in1_tot_format() -> None:
    content = """123 TOT
7.0  10  4
0.30 0 0
0.30 1 0
0.30 2 0
12.0
"""
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.in1", content)
        r = CaseFileParser.parse_in1(path)
        assert r["nbands"] == 123
        assert r["rkmax"] == 7.0
        assert r["lmax"] == 10
        assert r["v_nmt"] == 4.0
        assert r["format_type"] == "TOT"
        assert r["gmax"] == 12.0


def test_parse_in1_wffil_format() -> None:
    content = """WFFIL
7.0  10  4
0.30 0 0
0.30 1 0
0.30 2 0
14.0
"""
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.in1", content)
        r = CaseFileParser.parse_in1(path)
        assert r["format_type"] == "WFFIL"
        assert r["rkmax"] == 7.0
        assert r["gmax"] == 14.0


def test_parse_in1_tot_whitespace() -> None:
    content = "  42   TOT  \n  6.5  12  3  \n0.30 2 0\n"
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.in1", content)
        r = CaseFileParser.parse_in1(path)
        assert r["nbands"] == 42


def test_parse_in1_missing_file() -> None:
    r = CaseFileParser.parse_in1(Path("/nonexistent/case.in1"))
    assert r["nbands"] is None
    assert r["rkmax"] == 7.0


# ============================================================
# case.in2 parsing
# ============================================================


def test_parse_in2_fft_grid() -> None:
    content = """TOT
12.0
120 120 120 2.0 0
"""
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.in2", content)
        r = CaseFileParser.parse_in2(path)
        assert r["fft_nx"] == 120
        assert r["fft_ny"] == 120
        assert r["fft_nz"] == 120
        assert r["gmax"] == 12.0
        assert r["nmat_estimated"] > 0


def test_parse_in2_gmax_detection() -> None:
    content = """TOT
14.0
64 64 64 2.0 0
"""
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.in2", content)
        r = CaseFileParser.parse_in2(path)
        assert r["gmax"] == 14.0


def test_parse_in2_missing() -> None:
    r = CaseFileParser.parse_in2(Path("/nonexistent/case.in2"))
    assert r["fft_nx"] == 0
    assert r["fft_ny"] == 0
    assert r["fft_nz"] == 0


# ============================================================
# case.inm parsing — LDA+U
# ============================================================


def test_parse_inm_nmod1_simple() -> None:
    content = """1 2
1 2 0.30 0.00
2 2 0.35 0.02
"""
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.inm", content)
        ldau = CaseFileParser.parse_inm(path)
        assert ldau.file_present is True
        assert ldau.atoms == [1, 2]
        assert ldau.l_orbital == [2, 2]
        assert ldau.u_ry == [0.30, 0.35]
        assert ldau.j_ry == [0.00, 0.02]
        assert ldau.ueff_ry == [0.30, pytest.approx(0.33, abs=0.01)]
        assert ldau.double_counting == "AMF"


def test_parse_inm_with_double_counting() -> None:
    content = """1 2 1
1 2 0.30 0.00
2 2 0.35 0.02
"""
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.inm", content)
        ldau = CaseFileParser.parse_inm(path)
        assert ldau.double_counting == "FLL"


def test_parse_inm_empty() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.inm", "")
        ldau = CaseFileParser.parse_inm(path)
        assert ldau.file_present is True
        assert ldau.u_ry == []


def test_parse_inm_missing() -> None:
    ldau = CaseFileParser.parse_inm(Path("/nonexistent/case.inm"))
    assert ldau.file_present is False


# ============================================================
# case.scf parsing
# ============================================================


SCF_CONTENT = """:NMAT   :  2567
:FER  : F E R M I - ENERGY(GAUSS-SMEAR)=    0.45231
:ENE  : *  TOTAL ENERGY =   -1234.56789012
:LABEL4  : ITERATION  12
"""


def test_parse_scf_nmat() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.scf", SCF_CONTENT)
        r = CaseFileParser.parse_scf(path)
        assert r["nmat"] == 2567


def test_parse_scf_fermi() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.scf", SCF_CONTENT)
        r = CaseFileParser.parse_scf(path)
        assert r["fermi_energy_ry"] == 0.45231


def test_parse_scf_energy() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.scf", SCF_CONTENT)
        r = CaseFileParser.parse_scf(path)
        assert r["total_energy_ry"] < 0


def test_parse_scf_iterations() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.scf", SCF_CONTENT)
        r = CaseFileParser.parse_scf(path)
        assert r["scf_iterations"] == 12


# ============================================================
# case.struct parsing
# ============================================================


STRUCT_CONTENT = """NaCl rock salt
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


def test_parse_struct_atoms() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.struct", STRUCT_CONTENT)
        r = CaseFileParser.parse_struct(path)
        assert r["atoms"] == 8
        assert r["atoms_inequiv"] == 2


def test_parse_struct_spacegroup() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.struct", STRUCT_CONTENT)
        r = CaseFileParser.parse_struct(path)
        assert r["spacegroup"] == "225"


def test_parse_struct_volume() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.struct", STRUCT_CONTENT)
        r = CaseFileParser.parse_struct(path)
        assert r["volume_bohr3"] > 0


def test_parse_struct_missing() -> None:
    r = CaseFileParser.parse_struct(Path("/nonexistent/case.struct"))
    assert r["atoms"] == 0


# ============================================================
# case.klist parsing
# ============================================================


def test_parse_klist() -> None:
    content = "100\n"
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.klist", content)
        r = CaseFileParser.parse_klist(path)
        assert r["kpoints"] == 100


def test_parse_klist_fallback() -> None:
    content = """# header line
0.5 0.5 0.5 1.0
0.5 0.0 0.0 1.0
"""
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.klist", content)
        r = CaseFileParser.parse_klist(path)
        assert r["kpoints"] == 2


# ============================================================
# case.in0 parsing
# ============================================================


def test_parse_in0_rkmax() -> None:
    content = """TOT 13
RKMAX=8.0
"""
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.in0", content)
        r = CaseFileParser.parse_in0(path)
        assert r["rkmax"] == 8.0


def test_parse_in0_hybrid() -> None:
    content = """TOT
HYBR
"""
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.in0", content)
        r = CaseFileParser.parse_in0(path)
        assert r["is_hybrid"] is True


def test_parse_in0_no_hybrid() -> None:
    content = "TOT\n"
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.in0", content)
        r = CaseFileParser.parse_in0(path)
        assert r["is_hybrid"] is False


# ============================================================
# Flag detectors
# ============================================================


def test_detect_spin() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.inst", "Na\nAr 3\n3,2,2.0,0.0\n\nSPIN")
        assert CaseFileParser.detect_spin(path) is True


def test_detect_no_spin() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d) / "test.inst", "Na\nAr 3\n3,2,2.0,0.0\n")
        assert CaseFileParser.detect_spin(path) is False


def test_file_exists() -> None:
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "test.inso").write_text("", encoding="utf-8")
        assert CaseFileParser.file_exists(dp, "*.inso") is True
        assert CaseFileParser.file_exists(dp, "*.inorb") is False


# ============================================================
# CaseFileParser.parse_all integration test
# ============================================================


def _create_full_case_dir(base: Path, case: str) -> Path:
    case_dir = base / case
    case_dir.mkdir()
    _write(case_dir / f"{case}.struct", STRUCT_CONTENT)
    _write(case_dir / f"{case}.scf", SCF_CONTENT)
    _write(case_dir / f"{case}.in1", "123 TOT\n7.0 10 4\n0.30 0 0\n0.30 1 0\n0.30 2 0\n14.0\n")
    _write(case_dir / f"{case}.in2", "TOT\n14.0\n 120 120 120 2.0 0\n")
    _write(case_dir / f"{case}.in0", "TOT\nRKMAX=8.0\n")
    _write(case_dir / f"{case}.klist", "84\n")
    _write(case_dir / f"{case}.inst", "Na\nAr 3\n3,2,2.0,0.0\n\nSPIN")
    _write(case_dir / f"{case}.inso", "")
    _write(case_dir / f"{case}.inm", "1 2\n1 2 0.30 0.00\n2 2 0.35 0.02\n")
    return case_dir


def test_parse_all_complete() -> None:
    with tempfile.TemporaryDirectory() as d:
        case_dir = _create_full_case_dir(Path(d), "NaCl")
        parser = CaseFileParser(case_dir)
        data = parser.parse_all()

        assert data.case_name == "NaCl"
        assert data.atoms == 8
        assert data.nmat == 2567
        assert data.nbands == 123
        assert data.rkmax == 8.0  # .in0 takes precedence
        assert data.fft_nx == 120
        assert data.fft_ny == 120
        assert data.fft_nz == 120
        assert data.is_soc is True
        assert data.is_hybrid is False
        assert data.is_spin_polarized is True
        assert data.is_lda_u is True
        assert data.is_eece is False
        assert data.ldau.file_present is True
        assert data.ldau.u_ry == [0.30, 0.35]
        assert data.ldau.j_ry == [0.00, 0.02]
        assert data.scf_iterations == 12
        assert data.volume_bohr3 > 0


def test_parse_all_minimal() -> None:
    """Parser should not crash with minimal/empty case directory."""
    with tempfile.TemporaryDirectory() as d:
        case_dir = Path(d) / "empty_case"
        case_dir.mkdir()
        data = CaseFileParser(case_dir).parse_all()
        assert data.atoms == 0
        assert data.nmat == 0
        assert data.nbands is None
        assert data.is_soc is False


def test_convenience_function() -> None:
    with tempfile.TemporaryDirectory() as d:
        case_dir = _create_full_case_dir(Path(d), "FeO")
        data = parse_case_directory(case_dir)
        assert data.case_name == "FeO"
        assert data.atoms == 8


# ============================================================
# LDAUData
# ============================================================


def test_ldau_data_defaults() -> None:
    ldau = LDAUData()
    assert ldau.u_ry == []
    assert ldau.double_counting == "AMF"
    assert ldau.file_present is False
