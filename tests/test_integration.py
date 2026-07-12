"""Integration tests — end-to-end pipeline with realistic WIEN2k case directories."""

import tempfile
from pathlib import Path

from wien2k_gen.backends.wien2k import Wien2kBackend
from wien2k_gen.core.case_parser import CaseFileParser


def _create_case(base: Path, name: str, nmat=2567, nbands=256, atoms=8,
                 kpoints=84, rkmax=7.0, spin=False, soc=False, ldau=False):
    """Create a realistic WIEN2k case directory with all input files."""
    casedir = base / name
    casedir.mkdir()

    (casedir / f"{name}.struct").write_text("""NaCl rocksalt
H   LATTICE,NONEQUIV.ATOMS:  2 225 Fm-3m
MODE OF CALC=RELA unit=bohr
9.5 9.5 9.5 90.0 90.0 90.0
ATOM   1: X=0.00000000 Y=0.00000000 Z=0.00000000
          MULT= 4          ISPLIT= 8
Na1        NPT=  781  R0=.000010000 RMT=    2.50000   Z:  11.0
ATOM   2: X=0.50000000 Y=0.50000000 Z=0.50000000
          MULT= 4          ISPLIT= 8
Cl1        NPT=  781  R0=.000010000 RMT=    2.50000   Z:  17.0
""")

    (casedir / f"{name}.scf").write_text(f"""\
:NMAT   :  {nmat}
:FER  : F E R M I - ENERGY(GAUSS-SMEAR)=    0.30000
:ENE  : *  TOTAL ENERGY =   -1234.56789012
:LABEL4  : ITERATION  12
""")

    (casedir / f"{name}.in1").write_text(f"""\
{nbands} TOT
{rkmax}  10  4
0.30 0 0
0.30 1 0
0.30 2 0
12.0
""")

    (casedir / f"{name}.in2").write_text("""\
TOT
14.0
 100 100 100 2.0 0
""")

    (casedir / f"{name}.in0").write_text(f"""\
TOT  13
RKMAX={rkmax}
""")

    (casedir / f"{name}.klist").write_text(f"{kpoints}\n")

    if spin:
        (casedir / f"{name}.inst").write_text("Na\nAr 3\n3,2,2.0,0.0\n\nSPIN")
    if soc:
        (casedir / f"{name}.inso").write_text("")
    if ldau:
        (casedir / f"{name}.inorb").write_text("")
        (casedir / f"{name}.inm").write_text("1 2\n1 2 0.30 0.00\n2 2 0.35 0.02\n")

    return casedir


class TestEndToEndPipeline:
    """End-to-end: parse case → detect hardware → build suggestion → generate .machines."""

    def test_standard_scf_case(self):
        with tempfile.TemporaryDirectory() as d:
            casedir = _create_case(Path(d), "NaCl")
            parser = CaseFileParser(casedir)
            data = parser.parse_all()

            assert data.case_name == "NaCl"
            assert data.atoms == 8
            assert data.nmat == 2567
            assert data.nbands == 256
            assert data.rkmax == 7.0

    def test_wien2k_backend_detect_problem_size(self):
        with tempfile.TemporaryDirectory() as d:
            casedir = _create_case(Path(d), "test")
            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(casedir)
                backend = Wien2kBackend()
                result = backend.detect_problem_size()
                assert result["atoms"] == 8
                assert result["nmat"] == 2567
                assert result["nbands"] == 256
                assert result["rkmax"] == 7.0
            finally:
                os.chdir(old_cwd)

    def test_spin_detection_through_backend(self):
        with tempfile.TemporaryDirectory() as d:
            casedir = _create_case(Path(d), "FeO", spin=True)
            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(casedir)
                backend = Wien2kBackend()
                result = backend.detect_problem_size()
                assert result.get("is_spin_polarized") is True
                assert "runsp_lapw" in str(result.get("exec_command", ""))
            finally:
                os.chdir(old_cwd)

    def test_soc_detection_through_backend(self):
        with tempfile.TemporaryDirectory() as d:
            casedir = _create_case(Path(d), "Pt", spin=True, soc=True)
            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(casedir)
                backend = Wien2kBackend()
                result = backend.detect_problem_size()
                assert result.get("is_spin_polarized") is True
                assert result.get("is_soc") is True
                assert "runsp_lapw" in str(result.get("exec_command", ""))
                assert "-so" in str(result.get("exec_command", ""))
            finally:
                os.chdir(old_cwd)

    def test_lda_u_detection_through_backend(self):
        with tempfile.TemporaryDirectory() as d:
            casedir = _create_case(Path(d), "NiO", ldau=True)
            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(casedir)
                backend = Wien2kBackend()
                result = backend.detect_problem_size()
                assert result.get("is_lda_u") is True
                assert "-orbc" in str(result.get("exec_command", ""))
                ldau_u = result.get("_ldau_u_ry", [])
                assert len(ldau_u) == 2
                assert 0.25 < ldau_u[0] < 0.40
            finally:
                os.chdir(old_cwd)


class TestCompletePipeline:
    """Full pipeline: case → topology → suggestion → .machines content."""

    def test_build_machines_lines(self):
        """Verify .machines line generation is structurally correct."""
        with tempfile.TemporaryDirectory() as d:
            casedir = _create_case(Path(d), "Si", nmat=4500, nbands=200, atoms=2, kpoints=64)
            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(casedir)
                backend = Wien2kBackend()

                from wien2k_gen.core.topology import Topology
                topo = Topology(
                    nodes=["node01", "node02"],
                    cores_per_node=[32, 32],
                    env_type="slurm",
                )

                lines = backend._build_machines_lines(
                    topo,
                    {"mode": "kpoint", "recommended_total_cores": 32,
                     "omp_threads_per_rank": 1, "mpi_ranks_per_node": [16, 16],
                     "cores_per_node": [16, 16], "vector_split_active": False},
                )
                assert len(lines) > 0
                assert any("lapw0" in line for line in lines) or any(":1:" in line for line in lines)
            finally:
                os.chdir(old_cwd)
