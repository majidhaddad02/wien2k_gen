"""
WIEN2k Input File Parser (case.in1, case.in2, case.inm, etc.)

Extracts physical parameters from all WIEN2k input files needed for
memory estimation, parallelization strategy selection, and resource planning.

Parsed files and their key contributions:
    case.in1   → nbands (TOT/WFFIL), GMAX, RKMAX, LMAX
    case.in2   → FFT grid (nx, ny, nz), GMAX, TETRA flag
    case.inm   → LDA+U: U, J, double-counting per atom
    case.in0   → RKMAX, HYBR (hybrid functional flag)
    case.scf   → NMAT (exact basis set size), Fermi energy, SCF iterations
    case.struct → atoms, volume, lattice vectors, spacegroup
    case.klist  → kpoints, k-point type

References:
    Blaha et al. (2020) J. Chem. Phys. 152, 074101
    WIEN2k Usersguide Sections 4.1-4.5, 6.1, 7.3
    Cebrián et al. (2015) Comput. Phys. Commun. 201, 85-99
"""

from __future__ import annotations

import contextlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "CaseData",
    "CaseFileParser",
    "LDAUData",
    "check_struct_quality",
    "parse_case_directory",
]


@dataclass
class LDAUData:
    """LDA/U calculation parameters extracted from case.inm / case.inorb."""
    u_ry: list[float] = field(default_factory=list)
    j_ry: list[float] = field(default_factory=list)
    ueff_ry: list[float] = field(default_factory=list)
    l_orbital: list[int] = field(default_factory=list)
    atoms: list[int] = field(default_factory=list)
    double_counting: str = "AMF"
    file_present: bool = False


@dataclass
class CaseData:
    """Complete set of physical parameters extracted from a WIEN2k case directory."""
    case_name: str = ""
    atoms: int = 0
    atoms_inequiv: int = 0
    kpoints: int = 0
    nmat: int = 0
    nbands: int | None = None
    rkmax: float = 7.0
    lmax: int = 10
    v_nmt: float = 4.0
    gmax: float = 12.0
    fft_nx: int = 0
    fft_ny: int = 0
    fft_nz: int = 0
    is_soc: bool = False
    is_hybrid: bool = False
    is_spin_polarized: bool = False
    is_lda_u: bool = False
    is_eece: bool = False
    has_forces: bool = False
    ldau: LDAUData = field(default_factory=LDAUData)
    volume_bohr3: float = 0.0
    lattice_vectors: list[tuple[float, ...]] = field(default_factory=list)
    scf_iterations: int = 0
    fermi_energy_ry: float = 0.0
    total_energy_ry: float = 0.0
    wien2k_version: str = ""


class CaseFileParser:
    """
    Robust parser for all WIEN2k input files in a case directory.

    Usage:
        parser = CaseFileParser(Path("/path/to/case"))
        data = parser.parse_all()

        # Or parse individual files:
        nmat = CaseFileParser.parse_scf(Path("case.scf"))
        ldau = CaseFileParser.parse_inm(Path("case.inm"))
    """

    def __init__(self, case_dir: Path | None = None) -> None:
        if case_dir is None:
            case_dir = Path.cwd()
        case_dir = Path(case_dir)
        self._case_name: str | None = None
        if case_dir.is_file():
            if case_dir.suffix == ".struct":
                self._case_name = case_dir.stem
            self.case_dir = case_dir.parent
        else:
            self.case_dir = case_dir

    @property
    def case_name(self) -> str:
        if self._case_name is None:
            struct_files = sorted(self.case_dir.glob("*.struct"))
            if struct_files:
                self._case_name = struct_files[0].stem
            else:
                self._case_name = ""
        return self._case_name

    def _read_optional(self, glob_pat: str) -> tuple[Path, str] | None:
        files = sorted(self.case_dir.glob(glob_pat))
        if not files:
            return None
        try:
            content = files[0].read_text(encoding="utf-8", errors="replace")
            return (files[0], content)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # case.in1
    # ------------------------------------------------------------------

    @staticmethod
    def parse_in1(filepath: Path) -> dict[str, Any]:  # noqa: C901
        result: dict[str, Any] = {
            "nbands": None, "rkmax": 7.0, "lmax": 10,
            "v_nmt": 4.0, "gmax": 12.0, "format_type": "unknown",
        }
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return result

        lines = [ln.strip() for ln in content.splitlines()]
        if not lines:
            return result

        # Line 1: WFFIL | TOT | WF  [nbands TOT if old]  [nk / emax if WFFIL]
        first = lines[0]
        if "TOT" in first.upper() and not first.upper().startswith("WFFIL"):
            result["format_type"] = "TOT"
            parts = first.split()
            for p in parts:
                if p.isdigit():
                    result["nbands"] = int(p)
                    break
            if result["nbands"] is None:
                for i in range(len(parts) - 1):
                    if parts[i].isdigit() and parts[i + 1].upper() == "TOT":
                        result["nbands"] = int(parts[i])
                        break
        elif "WFFIL" in first.upper():
            result["format_type"] = "WFFIL"

        # Line 2: RKMAX  LMAX  V-NMT [/ V-NS]
        rkmax_lmax_found = False
        for line in lines[1:]:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rk = float(parts[0])
                    lm = int(parts[1])
                    vn = float(parts[2])
                    if rk > 0 and lm > 0:
                        result["rkmax"] = rk
                        result["lmax"] = lm
                        result["v_nmt"] = vn
                        rkmax_lmax_found = True
                        break
                except (ValueError, IndexError):
                    logger.debug("Suppressed exception in parse_in1()", exc_info=True)
            if rkmax_lmax_found:
                break

        # GMAX: appears as a single float on a line after the per-l QN block
        # Pattern: float value >= 4.0 on a line by itself after all QN lines
        gmax_candidates: list[float] = []
        in_qn_block = False
        qn_block_ended = False
        for line in lines[3:]:
            parts = line.split()
            if len(parts) == 1:
                try:
                    val = float(parts[0])
                    if (val >= 4.0 and qn_block_ended) or (val >= 4.0 and not in_qn_block):
                        gmax_candidates.append(val)
                except ValueError:
                    logger.debug("Suppressed exception in parse_in1()", exc_info=True)
            elif len(parts) == 2 and parts[0] != "K-VECTORS":
                try:
                    int(parts[0])
                    int(parts[1])
                    in_qn_block = True
                except ValueError:
                    if in_qn_block:
                        qn_block_ended = True
            elif in_qn_block and len(parts) > 2:
                pass
            else:
                if in_qn_block:
                    qn_block_ended = True

        if gmax_candidates:
            result["gmax"] = max(gmax_candidates)

        # WFFIL: nbands from nk/emax line (format: "number  number" after WFFIL)
        if result["nbands"] is None and result["format_type"] == "WFFIL":
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        result["nbands"] = int(parts[0])
                        break
                    except ValueError:
                        logger.debug("Suppressed exception in parse_in1()", exc_info=True)

        return result

    # ------------------------------------------------------------------
    # case.in2
    # ------------------------------------------------------------------

    @staticmethod
    def parse_in2(filepath: Path) -> dict[str, Any]:  # noqa: C901
        result: dict[str, Any] = {
            "fft_nx": 0, "fft_ny": 0, "fft_nz": 0,
            "gmax": 12.0, "tetra_method": False, "nmat_estimated": 0,
        }
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return result

        lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if not lines:
            return result

        # Line 1: TOT / FOR / QTL / FERMI
        if lines[0].upper().startswith("TETRA"):
            result["tetra_method"] = True

        # Look for GMAX (line 2 in standard format, float >= 4)
        for _i, line in enumerate(lines[:5]):
            parts = line.split()
            if len(parts) == 1:
                try:
                    val = float(parts[0])
                    if 4.0 <= val <= 30.0:
                        result["gmax"] = val
                        break
                except ValueError:
                    logger.debug("Suppressed exception in parse_in2()", exc_info=True)

        # FFT grid: line after GMAX (or line 3), three ints (may be followed by float)
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    nx = int(parts[0])
                    ny = int(parts[1])
                    nz = int(parts[2])
                    if nx > 0 and ny > 0 and nz > 0 and nx < 10000 and ny < 10000 and nz < 10000:
                        result["fft_nx"] = nx
                        result["fft_ny"] = ny
                        result["fft_nz"] = nz
                        break
                except ValueError:
                    logger.debug("Suppressed exception in parse_in2()", exc_info=True)

        if result["fft_nx"] > 0:
            fft_total = result["fft_nx"] * result["fft_ny"] * result["fft_nz"]
            result["nmat_estimated"] = max(100, int(fft_total ** (1.0 / 3.0) * 1.1))

        return result

    # ------------------------------------------------------------------
    # case.inm — LDA+U Hubbard parameters (NEW)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_inm(filepath: Path) -> LDAUData:  # noqa: C901
        """Parse LDA+U parameters from case.inm.

        WIEN2k .inm format (Blaha et al. 2020, Usersguide Section 6.1):
            Line 1:  [nmod]   [natorb]
            Line 2+: atom  l  U  J  [U_alpha  J_alpha  U_beta  J_beta]
            or (v19+ extended format):
            Line 1:  [nmod] [natorb] [ldc] [natorb2]
            Line 2+: atom s_o U J [U_alpha J_alpha ...]

        Returns LDAUData with U, J, Ueff for each correlated atom.
        """
        ldau = LDAUData()

        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ldau

        ldau.file_present = True

        lines = [ln.strip() for ln in content.splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]

        if not lines:
            return ldau

        # Header line: nmod [natorb [ldc]]
        header_parts = lines[0].split()
        try:
            header_ints = [int(p) for p in header_parts if p.lstrip("-").isdigit()]
        except ValueError:
            header_ints = []

        nmod = header_ints[0] if len(header_ints) >= 1 else 0
        if len(header_ints) >= 2:
            pass  # natorb — reserved
        if nmod not in (0, 1, 2, 3, 4):
            nmod = 1

        if len(header_ints) >= 3:
            ldc = header_ints[2]
            ldau.double_counting = {0: "AMF", 1: "FLL", 2: "SIC"}.get(ldc, "AMF")

        # Parse per-atom U/J lines
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                try:
                    nums = [float(p) for p in parts[:6]]
                except ValueError:
                    continue

            if len(nums) < 2:
                continue

            atom_idx = int(nums[0])
            if atom_idx == 0:
                continue  # end marker

            ldau.atoms.append(atom_idx)

            if nmod == 1:
                if len(nums) >= 3:
                    l_val = int(nums[1])
                    u = nums[2]
                    j = nums[3] if len(nums) >= 4 else 0.0
                    ldau.l_orbital.append(l_val)
                    ldau.u_ry.append(u)
                    ldau.j_ry.append(j)
                    ldau.ueff_ry.append(u - j if u >= j else u)
                elif len(nums) >= 2:
                    u = nums[1]
                    ldau.l_orbital.append(2)  # default d
                    ldau.u_ry.append(u)
                    ldau.j_ry.append(0.0)
                    ldau.ueff_ry.append(u)
            else:
                if len(nums) >= 4:
                    u = nums[2]
                    j = nums[3]
                    ldau.l_orbital.append(2)
                    ldau.u_ry.append(u)
                    ldau.j_ry.append(j)
                    ldau.ueff_ry.append(u - j if u >= j else u)
                elif len(nums) >= 2:
                    u = nums[1]
                    ldau.l_orbital.append(2)
                    ldau.u_ry.append(u)
                    ldau.j_ry.append(0.0)
                    ldau.ueff_ry.append(u)

        return ldau

    # ------------------------------------------------------------------
    # case.scf
    # ------------------------------------------------------------------

    @staticmethod
    def parse_scf(filepath: Path) -> dict[str, Any]:
        result: dict[str, Any] = {
            "nmat": 0, "fermi_energy_ry": 0.0,
            "total_energy_ry": 0.0, "scf_iterations": 0,
        }
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return result

        # :NMAT
        m = re.search(r':NMAT\s*:?\s*(\d+)', content)
        if m:
            result["nmat"] = int(m.group(1))

        # :FER (Fermi energy)
        m = re.search(r':FER\s*:.*=\s*([\d.E+\-]+)', content)
        if m:
            with contextlib.suppress(ValueError):
                result["fermi_energy_ry"] = float(m.group(1))

        # :ENE (Total energy)
        m = re.search(r':ENE\s*:.*=\s*([\-\d.E+\-]+)', content)
        if m:
            with contextlib.suppress(ValueError):
                result["total_energy_ry"] = float(m.group(1))

        # :ITER (SCF iterations)
        m = re.search(r':LABEL\d*\s*:\s*ITERATION\s+(\d+)', content)
        if m:
            result["scf_iterations"] = int(m.group(1))

        return result

    # ------------------------------------------------------------------
    # case.struct
    # ------------------------------------------------------------------

    @staticmethod
    def parse_struct(filepath: Path) -> dict[str, Any]:  # noqa: C901
        result: dict[str, Any] = {
            "atoms": 0, "atoms_inequiv": 0,
            "volume_bohr3": 0.0, "lattice_vectors": [],
            "spacegroup": "",
        }
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return result

        lines = content.splitlines()

        # NONEQUIV.ATOMS
        m = re.search(r'NONEQUIV\.ATOMS\s*:\s*(\d+)', content, re.IGNORECASE)
        if m:
            nat_inequiv = int(m.group(1))
            result["atoms_inequiv"] = nat_inequiv
            mult_matches = re.findall(r'MULT\s*=\s*(\d+)', content, re.IGNORECASE)
            if mult_matches and len(mult_matches) >= nat_inequiv:
                result["atoms"] = sum(int(m) for m in mult_matches[:nat_inequiv])
            else:
                result["atoms"] = nat_inequiv
        else:
            mult_matches = re.findall(r'MULT\s*=\s*(\d+)', content, re.IGNORECASE)
            if mult_matches:
                result["atoms"] = sum(int(m) for m in mult_matches)
            else:
                atom_pat = re.compile(r'^\s*ATOM\s*[-\d]+:', re.IGNORECASE)
                result["atoms"] = sum(1 for ln in lines if atom_pat.match(ln))

        # Spacegroup
        m = re.search(r'(\d+)\s+(I|P|F|C|R|A|B)[-\w]*\s*(?:RELA|NONE)?', content)
        if m:
            result["spacegroup"] = m.group(0).strip().split()[0]

        # Lattice vectors (line 4 after header, either 6-param or 3-vector format)
        try:
            # Find the lattice parameter line (6 floats = a,b,c,alpha,beta,gamma)
            # which is typically line 4 of the struct file
            lattice_line = None
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) >= 6:
                    floats = []
                    for p in parts[:6]:
                        v = try_float(p)
                        if v is None:
                            break
                        floats.append(v)
                    else:
                        if len(floats) == 6:
                            lattice_line = i
                            break
            if lattice_line is not None:
                parts = lines[lattice_line].strip().split()
                a, b, c = float(parts[0]), float(parts[1]), float(parts[2])
                alpha, beta, gamma = (float(parts[3]), float(parts[4]), float(parts[5]))
                import math
                alpha_r, beta_r, gamma_r = (math.radians(alpha), math.radians(beta), math.radians(gamma))
                # Volume = a*b*c * sqrt(1 - cos^2 a - cos^2 B - cos^2 y + 2cos a cos B cos y)
                ca, cb, cg = math.cos(alpha_r), math.cos(beta_r), math.cos(gamma_r)
                vol = a * b * c * math.sqrt(1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg)
                result["volume_bohr3"] = vol
                result["lattice_vectors"] = [(a, 0.0, 0.0)]
        except Exception:
            logger.debug("Suppressed exception in parse_struct()", exc_info=True)

        return result

    # ------------------------------------------------------------------
    # case.klist
    # ------------------------------------------------------------------

    @staticmethod
    def parse_klist(filepath: Path) -> dict[str, Any]:
        result: dict[str, int] = {"kpoints": 0}
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return result

        lines = [ln.strip() for ln in content.splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]

        if not lines:
            return result

        parts = lines[0].split()
        if parts and parts[0].isdigit():
            result["kpoints"] = int(parts[0])
        elif len(lines) >= 1:
            result["kpoints"] = len(lines)

        return result

    # ------------------------------------------------------------------
    # case.in0 / case.in0_st / case.in0_grr
    # ------------------------------------------------------------------

    @staticmethod
    def parse_in0(filepath: Path) -> dict[str, Any]:
        result: dict[str, Any] = {"rkmax": 7.0, "is_hybrid": False}
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return result

        m = re.search(r'RKMAX\s*=\s*([\d.]+)', content)
        if m:
            result["rkmax"] = float(m.group(1))

        if re.search(r'\bHYBR', content, re.IGNORECASE):
            result["is_hybrid"] = True

        return result

    # ------------------------------------------------------------------
    # Additional flag detectors
    # ------------------------------------------------------------------

    @staticmethod
    def detect_spin(filepath: Path) -> bool:
        """Check if case.inst contains SPIN keyword (spin-polarized calculation)."""
        try:
            return "SPIN" in filepath.read_text(encoding="utf-8", errors="replace").upper()
        except Exception:
            return False

    @staticmethod
    def file_exists(case_dir: Path, pattern: str) -> bool:
        return len(list(case_dir.glob(pattern))) > 0

    # ------------------------------------------------------------------
    # Parse all
    # ------------------------------------------------------------------

    def parse_all(self) -> CaseData:  # noqa: C901
        """Parse all available case.* files and return a complete CaseData."""
        data = CaseData(case_name=self.case_name)

        # .struct
        r = self._read_optional("*.struct")
        if r is not None:
            s = self.parse_struct(r[0])
            data.atoms = int(s.get("atoms", 0) or 0)
            data.atoms_inequiv = int(s.get("atoms_inequiv", 0) or 0)
            data.volume_bohr3 = float(s.get("volume_bohr3", 0.0) or 0.0)
            data.lattice_vectors = s.get("lattice_vectors", []) or []

        # .klist
        r = self._read_optional("*.klist*")
        if r is not None:
            k = self.parse_klist(r[0])
            data.kpoints = k.get("kpoints", 0)

        # .scf
        r = self._read_optional("*.scf")
        if r is not None:
            s = self.parse_scf(r[0])
            data.nmat = int(s.get("nmat", 0) or 0)
            data.scf_iterations = int(s.get("scf_iterations", 0) or 0)
            data.fermi_energy_ry = float(s.get("fermi_energy_ry", 0.0) or 0.0)
            data.total_energy_ry = float(s.get("total_energy_ry", 0.0) or 0.0)

        # .in2
        r = self._read_optional("*.in2*")
        if r is not None:
            i2 = self.parse_in2(r[0])
            data.fft_nx = int(i2.get("fft_nx", 0) or 0)
            data.fft_ny = int(i2.get("fft_ny", 0) or 0)
            data.fft_nz = int(i2.get("fft_nz", 0) or 0)
            data.gmax = float(i2.get("gmax", 12.0) or 12.0)
            if data.nmat == 0:
                data.nmat = int(i2.get("nmat_estimated", 0) or 0)

        # .in1
        r = self._read_optional("*.in1*")
        if r is not None:
            i1 = self.parse_in1(r[0])
            data.nbands = int(i1.get("nbands", None) or 0) if i1.get("nbands") else None
            data.rkmax = float(i1.get("rkmax", 7.0) or 7.0)
            data.lmax = int(i1.get("lmax", 10) or 10)
            data.v_nmt = float(i1.get("v_nmt", 4.0) or 4.0)
            gmax1 = i1.get("gmax", 12.0)
            if isinstance(gmax1, (int, float)) and (gmax1 or 12.0) > data.gmax:
                data.gmax = float(gmax1)

        # .in0 / .in0_st / .in0_grr
        for pat in ["*.in0", "*.in0_st", "*.in0_grr"]:
            r = self._read_optional(pat)
            if r is not None:
                i0 = self.parse_in0(r[0])
                if not data.rkmax or data.rkmax == 7.0:
                    data.rkmax = float(i0.get("rkmax", 7.0) or 7.0)
                if i0.get("is_hybrid"):
                    data.is_hybrid = True

        # .inc (hybrid functional)
        r = self._read_optional("*.inc")
        if r is not None and re.search(r'\bHYBR', r[1], re.IGNORECASE):
            data.is_hybrid = True

        # .inst (spin polarization)
        r = self._read_optional("*.inst")
        if r is not None:
            data.is_spin_polarized = self.detect_spin(r[0])

        # .inso
        data.is_soc = self.file_exists(self.case_dir, "*.inso")

        # .inorb → LDA+U
        if self.file_exists(self.case_dir, "*.inorb"):
            data.is_lda_u = True

        # .inm → LDA+U parameters
        r = self._read_optional("*.inm")
        if r is not None:
            data.ldau = self.parse_inm(r[0])
            if data.ldau.file_present:
                data.is_lda_u = True

        # .ineece
        data.is_eece = self.file_exists(self.case_dir, "*.ineece")

        # Forces: detected via .in2 TOT/FOR flag or presence of -fc flag files
        if self.file_exists(self.case_dir, "*.in2"):
            r = self._read_optional("*.in2*")
            if r is not None and "FOR" in r[1].upper():
                data.has_forces = True

        # WIEN2k version
        import os
        wienroot = os.environ.get("WIENROOT")
        if wienroot:
            version_file = Path(wienroot, "VERSION")
            if version_file.exists():
                with contextlib.suppress(Exception):
                    data.wien2k_version = version_file.read_text().strip().split()[0]

        # nbands fallback
        if data.nbands is None and data.nmat > 0:
            data.nbands = max(10, data.nmat // 2)

        return data


def parse_case_directory(path: Path | None = None) -> CaseData:
    """Convenience function: parse all WIEN2k input files in a directory."""
    return CaseFileParser(path).parse_all()


def detect_wien2k_version() -> str:  # noqa: C901
    """Detect WIEN2k version from WIENROOT environment and installed files.

    WIEN2k version history and key changes:
      19.x — ELPA support introduced, band parallelization improvements
      21.x — Improved hybrid functionals, GPU experimental support
      23.x — GPU acceleration (experimental), improved SOC performance
      24.x — Enhanced fine_grain parallelization, better NUMA support

    Returns version string like "24.1" or "unknown".
    """
    import subprocess as _sp

    wienroot = os.environ.get("WIENROOT", "")
    if not wienroot:
        return "unknown"

    candidates = [
        Path(wienroot) / "VERSION",
        Path(wienroot) / "WIEN2k_VERSION",
        Path(wienroot) / "version.txt",
    ]
    for vf in candidates:
        if vf.exists():
            content = vf.read_text().strip()
            m = re.search(r'(\d+\.\d+)', content)
            if m:
                return m.group(1)

    try:
        result = _sp.run(
            ["x_lapw", "--version"], capture_output=True, text=True, timeout=5,
        )
        m = re.search(r'(\d+\.\d+)', result.stdout + result.stderr)
        if m:
            return m.group(1)
    except Exception:
        logger.debug("Suppressed exception in detect_wien2k_version()", exc_info=True)

    try:
        lv = Path(wienroot) / "SRC_lapw1" / "lapw1.F"
        if lv.exists():
            content = lv.read_text()
            for line in content.split('\n')[:20]:
                m = re.search(r'version.*?(\d+\.\d+)', line, re.IGNORECASE)
                if m:
                    return m.group(1)
    except Exception:
        logger.debug("Suppressed exception in detect_wien2k_version()", exc_info=True)

    return "unknown"


_VERSION_CAPABILITIES = {
    "19": {"elpa": True, "band_par": True, "gpu": False, "fine_grain": "basic"},
    "21": {"elpa": True, "band_par": True, "gpu": "experimental", "fine_grain": "basic"},
    "23": {"elpa": True, "band_par": True, "gpu": "experimental", "fine_grain": "enhanced"},
    "24": {"elpa": True, "band_par": True, "gpu": "supported", "fine_grain": "full"},
}


def wien2k_supports(capability: str) -> bool:
    """Check if detected WIEN2k version supports a specific capability."""
    version = detect_wien2k_version()
    if version == "unknown":
        return True
    major = version.split('.')[0]
    caps = _VERSION_CAPABILITIES.get(major, {})
    val = caps.get(capability)
    if isinstance(val, bool):
        return val
    return val is not None


# ------------------------------------------------------------------
# Tiny 3D vector for volume calculation (no numpy dependency)
# ------------------------------------------------------------------

class Vector:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def dot(self, other: Vector) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector) -> Vector:
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


def try_float(s: str) -> float | None:
    """Try to parse a float, returning None on failure."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def check_struct_quality(struct_path: Path) -> dict[str, Any]:  # noqa: C901
    """Check WIEN2k .struct file for common issues.

    Ref: Blaha et al., WIEN2k User Guide — struct preparation.
    Warns about:
      - RMT sphere overlaps (>10% overlap triggers strong warning)
      - Very small RMT for light elements (O, F, N)
      - Zone symbol / Wyckoff position warnings

    Returns dict with:
        warnings: List[str]
        errors: List[str]
        rmt_data: List[dict] — per-atom RMT/Z/position info
    """
    result: dict[str, Any] = {"warnings": [], "errors": [], "rmt_data": []}

    try:
        content = struct_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        result["errors"].append(f"Cannot read struct file: {struct_path}")
        return result

    lines = content.splitlines()

    # Extract lattice constants
    a, b, c = 1.0, 1.0, 1.0
    alpha, beta, gamma = 90.0, 90.0, 90.0
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 6:
            vals = [try_float(p) for p in parts[:6]]
            if all(v is not None for v in vals):
                a, b, c = vals[0], vals[1], vals[2]
                alpha, beta, gamma = vals[3], vals[4], vals[5]
                break

    alpha_r, beta_r, gamma_r = (
        math.radians(alpha), math.radians(beta), math.radians(gamma))

    # Fractional → Cartesian conversion
    ca, cb, cg = math.cos(alpha_r), math.cos(beta_r), math.cos(gamma_r)
    sg = math.sin(gamma_r)
    cart_matrix = (
        (a, b * cg, c * cb),
        (0.0, b * sg, c * (ca - cb * cg) / sg if sg > 1e-12 else 0.0),
        (0.0, 0.0, c * math.sqrt(1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg) / sg
         if sg > 1e-12 else c),
    )

    def frac_to_cart(x, y, z):
        vx = cart_matrix[0][0] * x + cart_matrix[0][1] * y + cart_matrix[0][2] * z
        vy = cart_matrix[1][0] * x + cart_matrix[1][1] * y + cart_matrix[1][2] * z
        vz = cart_matrix[2][0] * x + cart_matrix[2][1] * y + cart_matrix[2][2] * z
        return (vx, vy, vz)

    # Parse ATOM entries
    atom_positions = []
    atom_rmts = []
    atom_zs = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'ATOM\s+\d+:\s+X=\s*([\d\.\-]+)\s+Y=\s*([\d\.\-]+)\s+Z=\s*([\d\.\-]+)', line, re.IGNORECASE)
        if m:
            x = float(m.group(1))
            y = float(m.group(2))
            z = float(m.group(3))
            atom_positions.append((x, y, z))
            # Next lines: MULT, then atom data with RMT
            if i + 2 < len(lines):
                data_line = lines[i + 2].strip()
                rmt_match = re.search(r'RMT\s*=\s*([\d\.]+)', data_line, re.IGNORECASE)
                z_match = re.search(r'Z\s*:\s*([\d\.]+)', data_line, re.IGNORECASE)
                atom_rmts.append(float(rmt_match.group(1)) if rmt_match else 0.0)
                atom_zs.append(int(float(z_match.group(1))) if z_match else 0)
            else:
                atom_rmts.append(0.0)
                atom_zs.append(0)
        i += 1

    if not atom_positions:
        result["warnings"].append("No ATOM positions found in struct — file may be incomplete")
        return result

    for idx, (pos, rmt, z) in enumerate(zip(atom_positions, atom_rmts, atom_zs)):
        result["rmt_data"].append({
            "index": idx, "x": pos[0], "y": pos[1], "z": pos[2],
            "rmt": rmt, "charge": z,
        })

    # Check RMT overlaps between inequivalent atoms
    for i in range(len(atom_positions)):
        for j in range(i + 1, len(atom_positions)):
            pi = atom_positions[i]
            pj = atom_positions[j]
            rmt_sum = atom_rmts[i] + atom_rmts[j]
            if rmt_sum <= 0:
                continue

            cart_i = frac_to_cart(*pi)
            cart_j = frac_to_cart(*pj)
            dist = math.sqrt(
                (cart_i[0] - cart_j[0])**2 +
                (cart_i[1] - cart_j[1])**2 +
                (cart_i[2] - cart_j[2])**2
            )

            overlap = rmt_sum - dist
            overlap_pct = (overlap / dist) * 100 if dist > 0 else 100.0

            if overlap_pct > 30:
                result["errors"].append(
                    f"CRITICAL: RMT spheres overlap by {overlap_pct:.0f}% "
                    f"between atom {i} (RMT={atom_rmts[i]:.2f}) and atom {j} "
                    f"(RMT={atom_rmts[j]:.2f}). Reduce RMT to < 0.85xnearest-neighbor distance."
                )
            elif overlap_pct > 10:
                result["warnings"].append(
                    f"RMT spheres overlap by {overlap_pct:.0f}% between "
                    f"atom {i} and atom {j}. Consider reducing RMT to avoid linearization errors."
                )
            elif overlap_pct > 0:
                result["warnings"].append(
                    f"Marginal RMT overlap ({overlap_pct:.1f}%) between "
                    f"atom {i} and atom {j}. Monitor SCF convergence."
                )

    # Warn about small RMT for light hard elements (O, F, N)
    hard_z = {8, 9, 7}
    for idx, (rmt, z) in enumerate(zip(atom_rmts, atom_zs)):
        if z in hard_z and rmt < 1.4:
            result["warnings"].append(
                f"Atom {idx} (Z={z}) has very small RMT={rmt:.2f} bohr. "
                f"Hard potentials of light elements need RKMAX ≥ 7.0 for convergence."
            )

    # Wyckoff / symmetry heuristic
    if "MULT" in content and "NONEQUIV.ATOMS" in content:
        m = re.search(r'NONEQUIV\.ATOMS\s*:\s*\d+\s+(\d+)[-_]', content, re.IGNORECASE)
        if m:
            sg = int(m.group(1))
            if sg < 2:
                result["warnings"].append(
                    f"Spacegroup {sg}: triclinic — ensure exact Wyckoff positions "
                    f"to avoid symmetry breaking during SCF"
                )

    return result


# ===========================================================================
# Phase 2 — setrmt Algorithm (JCP 2020)
# ===========================================================================

def parse_crystal_structure(struct_path: Path) -> dict[str, Any]:
    """Parse WIEN2k .struct file for crystal structure data.

    Extracts:
        lattice: dict with a,b,c,alpha,beta,gamma (bohr, degrees)
        atoms: List[dict] — fractional coordinates, atomic number, RMT
        spacegroup: str
        num_atoms: int

    Returns empty dict on error.
    """
    try:
        content = struct_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}

    lines = content.splitlines()
    result: dict[str, Any] = {
        "lattice": {"a": 1.0, "b": 1.0, "c": 1.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        "atoms": [],
        "spacegroup": "",
        "num_atoms": 0,
    }

    # Lattice parameters (6-float line)
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 6:
            vals = [try_float(p) for p in parts[:6]]
            if all(v is not None for v in vals):
                result["lattice"] = {
                    "a": vals[0], "b": vals[1], "c": vals[2],
                    "alpha": vals[3], "beta": vals[4], "gamma": vals[5],
                }
                break

    # Spacegroup
    m = re.search(r'(\d+)\s+(I|P|F|C|R|A|B)[-\w]*\s*(?:RELA|NONE)?', content)
    if m:
        result["spacegroup"] = m.group(0).strip().split()[0]

    # ATOM entries
    atom_entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(
            r'ATOM\s+\d+:\s+X=\s*([\d\.\-]+)\s+Y=\s*([\d\.\-]+)\s+Z=\s*([\d\.\-]+)',
            line, re.IGNORECASE)
        if m:
            x = float(m.group(1))
            y = float(m.group(2))
            z = float(m.group(3))
            atom_entry = {"x": x, "y": y, "z": z, "rmt": 0.0, "z_num": 0}
            if i + 2 < len(lines):
                data_line = lines[i + 2].strip()
                rmt_match = re.search(r'RMT\s*=\s*([\d\.]+)', data_line, re.IGNORECASE)
                z_match = re.search(r'Z\s*:\s*([\d\.]+)', data_line, re.IGNORECASE)
                atom_entry["rmt"] = float(rmt_match.group(1)) if rmt_match else 0.0
                atom_entry["z_num"] = int(float(z_match.group(1))) if z_match else 0
            atom_entries.append(atom_entry)
        i += 1

    result["atoms"] = atom_entries
    result["num_atoms"] = len(atom_entries)

    return result


def calculate_nn_distances(structure: dict[str, Any]) -> dict[int, float]:  # noqa: C901
    """Calculate nearest-neighbor distances for all inequivalent atoms.

    Algorithm:
      1. Convert fractional to Cartesian coordinates
      2. Build 3x3x3 supercell for periodic images
      3. For each atom, find minimum distance to any other atom
      4. Exclude self-distance

    Returns dict: atom_index → nn_distance (bohr)
    """
    atoms = structure.get("atoms", [])
    if not atoms:
        return {}

    lat = structure.get("lattice", {})
    a = lat.get("a", 1.0)
    b = lat.get("b", 1.0)
    c = lat.get("c", 1.0)
    alpha = lat.get("alpha", 90.0)
    beta = lat.get("beta", 90.0)
    gamma = lat.get("gamma", 90.0)

    alpha_r, beta_r, gamma_r = (math.radians(alpha), math.radians(beta), math.radians(gamma))

    # Cartesian conversion matrix
    ca, cb, cg = math.cos(alpha_r), math.cos(beta_r), math.cos(gamma_r)
    sg = math.sin(gamma_r)
    cart_matrix = (
        (a, b * cg, c * cb),
        (0.0, b * sg, c * (ca - cb * cg) / sg if sg > 1e-12 else 0.0),
        (0.0, 0.0, c * math.sqrt(1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg) / sg if sg > 1e-12 else c),
    )

    # Convert fractional to Cartesian for all atoms
    atom_coords = []
    for atom in atoms:
        x = atom["x"]
        y = atom["y"]
        z = atom["z"]
        cx = cart_matrix[0][0] * x + cart_matrix[0][1] * y + cart_matrix[0][2] * z
        cy = cart_matrix[1][0] * x + cart_matrix[1][1] * y + cart_matrix[1][2] * z
        cz = cart_matrix[2][0] * x + cart_matrix[2][1] * y + cart_matrix[2][2] * z
        atom_coords.append((cx, cy, cz))

    # Build supercell (images -1, 0, 1 in each direction)
    images = []
    for idx, (cx, cy, cz) in enumerate(atom_coords):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    sc_x = cx + di * cart_matrix[0][0] + dj * cart_matrix[0][1] + dk * cart_matrix[0][2]
                    sc_y = cy + di * cart_matrix[1][0] + dj * cart_matrix[1][1] + dk * cart_matrix[1][2]
                    sc_z = cz + di * cart_matrix[2][0] + dj * cart_matrix[2][1] + dk * cart_matrix[2][2]
                    images.append((idx, sc_x, sc_y, sc_z))

    # Find nearest neighbor for each atom
    nn_distances: dict[int, float] = {}
    n_atoms = len(atom_coords)

    for i in range(n_atoms):
        min_dist = float("inf")
        ci_x, ci_y, ci_z = atom_coords[i]
        for idx_j, sj_x, sj_y, sj_z in images:
            if idx_j == i:
                continue
            dist = math.sqrt(
                (ci_x - sj_x) ** 2 + (ci_y - sj_y) ** 2 + (ci_z - sj_z) ** 2)
            if 0.001 < dist < min_dist:
                min_dist = dist
        nn_distances[i] = min_dist if min_dist < float("inf") else 0.0

    return nn_distances


def optimize_rmt(
    nn_distances: dict[int, float],
    reduction_factor: float = 0.95,
    min_rmt: float = 2.5,
    max_rmt: float = 4.0,
) -> dict[int, float]:
    """Calculate optimal RMT values from nearest-neighbor distances.

    Formula (Blaha et al., JCP 2020):
        RMT_optimal = reduction_factor x (nn_distance / 2)

    Constraints:
        - RMT ≥ min_rmt (2.5 a.u. for very light elements)
        - RMT ≤ max_rmt (4.0 a.u. for heavy elements)

    Returns dict: atom_index → optimal_rmt (bohr)
    """
    optimal = {}
    for idx, nn_dist in nn_distances.items():
        if nn_dist <= 0:
            continue
        rmt = reduction_factor * (nn_dist / 2.0)
        rmt = max(min_rmt, min(max_rmt, rmt))
        optimal[idx] = round(rmt, 3)
    return optimal


def check_rmt_overlaps(
    rmts: dict[int, float],
    structure: dict[str, Any],
    overlap_warning: float = 0.95,
    overlap_critical: float = 1.00,
) -> list[dict[str, Any]]:
    """Check RMT sphere overlaps between inequivalent atoms.

    Overlap = (RMT_i + RMT_j) / nn_distance_ij

    Thresholds:
        overlap > 1.00  → CRITICAL (spheres intersect)
        overlap > 0.95  → WARNING (marginal)
        overlap ≤ 0.95  → OK

    Returns list of overlap entries.
    """
    atoms = structure.get("atoms", [])
    calculate_nn_distances(structure)
    overlaps = []

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            rmt_i = rmts.get(i, atoms[i]["rmt"] if i < len(atoms) else 1.5)
            rmt_j = rmts.get(j, atoms[j]["rmt"] if j < len(atoms) else 1.5)

            atoms[i]["x"], atoms[i]["y"], atoms[i]["z"]
            atoms[j]["x"], atoms[j]["y"], atoms[j]["z"]

            lat = structure.get("lattice", {})
            a, b_val, c = lat.get("a", 1.0), lat.get("b", 1.0), lat.get("c", 1.0)
            alpha, beta, gamma = lat.get("alpha", 90.0), lat.get("beta", 90.0), lat.get("gamma", 90.0)

            ca, cb, cg = math.cos(math.radians(alpha)), math.cos(math.radians(beta)), math.cos(math.radians(gamma))
            sg = math.sin(math.radians(gamma))
            mx = (
                (a, b_val * cg, c * cb),
                (0.0, b_val * sg, c * (math.cos(math.radians(alpha)) - cb * cg) / sg if sg > 1e-12 else 0.0),
                (0.0, 0.0, c * math.sqrt(1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg) / sg if sg > 1e-12 else c),
            )

            ci_x = mx[0][0]*atoms[i]["x"] + mx[0][1]*atoms[i]["y"] + mx[0][2]*atoms[i]["z"]
            ci_y = mx[1][0]*atoms[i]["x"] + mx[1][1]*atoms[i]["y"] + mx[1][2]*atoms[i]["z"]
            ci_z = mx[2][0]*atoms[i]["x"] + mx[2][1]*atoms[i]["y"] + mx[2][2]*atoms[i]["z"]
            cj_x = mx[0][0]*atoms[j]["x"] + mx[0][1]*atoms[j]["y"] + mx[0][2]*atoms[j]["z"]
            cj_y = mx[1][0]*atoms[j]["x"] + mx[1][1]*atoms[j]["y"] + mx[1][2]*atoms[j]["z"]
            cj_z = mx[2][0]*atoms[j]["x"] + mx[2][1]*atoms[j]["y"] + mx[2][2]*atoms[j]["z"]

            dist = math.sqrt((ci_x - cj_x)**2 + (ci_y - cj_y)**2 + (ci_z - cj_z)**2)
            if dist <= 0:
                continue

            overlap_val = (rmt_i + rmt_j) / dist

            if overlap_val > overlap_critical:
                severity = "critical"
            elif overlap_val > overlap_warning:
                severity = "warning"
            else:
                continue

            overlaps.append({
                "atom_i": i,
                "atom_j": j,
                "rmt_i": rmt_i,
                "rmt_j": rmt_j,
                "distance": round(dist, 4),
                "overlap": round(overlap_val, 4),
                "severity": severity,
            })

    return overlaps


def recommend_final_rmt(
    optimal_rmts: dict[int, float],
    overlaps: list[dict[str, Any]],
    structure: dict[str, Any],
) -> dict[int, float]:
    """Adjust optimal RMT values to eliminate critical overlaps.

    For each critical or warning overlap, reduces both RMT values
    proportionally until overlap drops below 0.95.
    """
    final = dict(optimal_rmts)
    structure.get("atoms", [])

    for ov in overlaps:
        i, j = ov["atom_i"], ov["atom_j"]
        if ov["severity"] == "critical" or ov["overlap"] > 0.95:
            target = 0.94 * ov["distance"]
            reduction = target / (ov["rmt_i"] + ov["rmt_j"]) if (ov["rmt_i"] + ov["rmt_j"]) > 0 else 1.0
            new_i = ov["rmt_i"] * reduction
            new_j = ov["rmt_j"] * reduction
            final[i] = round(new_i, 3)
            final[j] = round(new_j, 3)

    return final


def generate_rmt_report(
    final_rmts: dict[int, float],
    overlaps: list[dict[str, Any]],
    nn_distances: dict[int, float],
    structure: dict[str, Any],
) -> str:
    """Generate human-readable RMT optimization report.

    Format:
        RMT Optimization Report
        =======================
        Atom  Element  NN_Dist(A)  Optimal_RMT  Final_RMT  Overlap
        1     Fe       2.87        1.36          1.35       0.94

        Warnings:
        - Atom 1: RMT reduced to avoid overlap with Atom 2
        - Atom 2: Small RMT, recommend RKMAX ≥ 7.0
    """
    atoms = structure.get("atoms", [])
    element_names = {
        1: "H", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na",
        12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K",
        20: "Ca", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co",
        28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
        38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 44: "Ru", 45: "Rh",
        46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 52: "Te",
        56: "Ba", 57: "La", 58: "Ce", 64: "Gd", 72: "Hf", 73: "Ta", 74: "W",
        75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 82: "Pb",
        83: "Bi", 90: "Th", 92: "U",
    }

    lines = [
        "RMT Optimization Report",
        "=======================",
        "Generated by forge (Blaha et al., JCP 2020)",
        "",
    ]

    # Header
    lines.append(f"{'Atom':<6} {'Elem':<6} {'NN(A)':<10} {'Opt_RMT':<10} {'Final_RMT':<10} {'Overlap':<10}")
    lines.append("-" * 56)

    for idx in sorted(final_rmts.keys()):
        atom = atoms[idx] if idx < len(atoms) else {}
        elem = element_names.get(atom.get("z_num", 0), f"X{atom.get('z_num', idx)}")
        nn_dist = nn_distances.get(idx, 0.0)
        nn_a = nn_dist * 0.529177  # bohr → Å
        opt_rmt = final_rmts.get(idx, 0.0)
        overlap_str = "-"
        for ov in overlaps:
            if ov["atom_i"] == idx or ov["atom_j"] == idx:
                overlap_str = f"{ov['overlap']:.2f}"
                break
        lines.append(f"{idx+1:<6} {elem:<6} {nn_a:<10.3f} {opt_rmt:<10.3f} {opt_rmt:<10.3f} {overlap_str:<10}")

    # Warnings
    warnings_list = []
    for ov in overlaps:
        if ov["severity"] == "critical":
            i, j = ov["atom_i"], ov["atom_j"]
            warnings_list.append(
                f"- Atom {i+1}: CRITICAL overlap ({ov['overlap']:.3f}) with Atom {j+1}. "
                f"RMT reduced from {ov['rmt_i']:.2f} → {final_rmts.get(i, ov['rmt_i']):.2f}"
            )

    for idx, rmt in final_rmts.items():
        if rmt < 2.5:
            warnings_list.append(
                f"- Atom {idx+1}: RMT very small ({rmt:.2f} a.u.), recommend RKMAX ≥ 7.0"
            )
        if rmt > 3.5:
            warnings_list.append(
                f"- Atom {idx+1}: RMT large ({rmt:.2f} a.u.), check for overlaps"
            )

    if warnings_list:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(warnings_list)

    return "\n".join(lines)
