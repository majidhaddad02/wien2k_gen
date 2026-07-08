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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "CaseData",
    "CaseFileParser",
    "LDAUData",
    "parse_case_directory",
]


@dataclass
class LDAUData:
    """LDA/U calculation parameters extracted from case.inm / case.inorb."""
    u_ry: List[float] = field(default_factory=list)
    j_ry: List[float] = field(default_factory=list)
    ueff_ry: List[float] = field(default_factory=list)
    l_orbital: List[int] = field(default_factory=list)
    atoms: List[int] = field(default_factory=list)
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
    nbands: Optional[int] = None
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
    lattice_vectors: List[Tuple[float, ...]] = field(default_factory=list)
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

    def __init__(self, case_dir: Optional[Path] = None) -> None:
        if case_dir is None:
            case_dir = Path.cwd()
        self.case_dir = Path(case_dir)
        self._case_name: Optional[str] = None

    @property
    def case_name(self) -> str:
        if self._case_name is None:
            struct_files = sorted(self.case_dir.glob("*.struct"))
            if struct_files:
                self._case_name = struct_files[0].stem
            else:
                self._case_name = ""
        return self._case_name

    def _read_optional(self, glob_pat: str) -> Optional[Tuple[Path, str]]:
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
    def parse_in1(filepath: Path) -> Dict[str, Any]:
        result: Dict[str, Any] = {
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
                    pass
            if rkmax_lmax_found:
                break

        # GMAX: appears as a single float on a line after the per-l QN block
        # Pattern: float value >= 4.0 on a line by itself after all QN lines
        gmax_candidates: List[float] = []
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
                    pass
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
                        pass

        return result

    # ------------------------------------------------------------------
    # case.in2
    # ------------------------------------------------------------------

    @staticmethod
    def parse_in2(filepath: Path) -> Dict[str, Any]:
        result: Dict[str, Any] = {
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
        for i, line in enumerate(lines[:5]):
            parts = line.split()
            if len(parts) == 1:
                try:
                    val = float(parts[0])
                    if 4.0 <= val <= 30.0:
                        result["gmax"] = val
                        break
                except ValueError:
                    pass

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
                    pass

        if result["fft_nx"] > 0:
            fft_total = result["fft_nx"] * result["fft_ny"] * result["fft_nz"]
            result["nmat_estimated"] = max(100, int(fft_total ** (1.0 / 3.0) * 1.1))

        return result

    # ------------------------------------------------------------------
    # case.inm — LDA+U Hubbard parameters (NEW)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_inm(filepath: Path) -> LDAUData:
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
                nums = [float(p) if "." in p or "e" in p.lower() else float(p)
                        for p in parts]
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
    def parse_scf(filepath: Path) -> Dict[str, Any]:
        result: Dict[str, Any] = {
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
            try:
                result["fermi_energy_ry"] = float(m.group(1))
            except ValueError:
                pass

        # :ENE (Total energy)
        m = re.search(r':ENE\s*:.*=\s*([\-\d.E+\-]+)', content)
        if m:
            try:
                result["total_energy_ry"] = float(m.group(1))
            except ValueError:
                pass

        # :ITER (SCF iterations)
        m = re.search(r':LABEL\d*\s*:\s*ITERATION\s+(\d+)', content)
        if m:
            result["scf_iterations"] = int(m.group(1))

        return result

    # ------------------------------------------------------------------
    # case.struct
    # ------------------------------------------------------------------

    @staticmethod
    def parse_struct(filepath: Path) -> Dict[str, Any]:
        result: Dict[str, Any] = {
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
                # Volume = a*b*c * sqrt(1 - cos²α - cos²β - cos²γ + 2cosα cosβ cosγ)
                ca, cb, cg = math.cos(alpha_r), math.cos(beta_r), math.cos(gamma_r)
                vol = a * b * c * math.sqrt(1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg)
                result["volume_bohr3"] = vol
                result["lattice_vectors"] = [(a, 0.0, 0.0)]
        except Exception:
            pass

        return result

    # ------------------------------------------------------------------
    # case.klist
    # ------------------------------------------------------------------

    @staticmethod
    def parse_klist(filepath: Path) -> Dict[str, Any]:
        result: Dict[str, int] = {"kpoints": 0}
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
    def parse_in0(filepath: Path) -> Dict[str, Any]:
        result: Dict[str, Any] = {"rkmax": 7.0, "is_hybrid": False}
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

    def parse_all(self) -> CaseData:
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
        if r is not None:
            if re.search(r'\bHYBR', r[1], re.IGNORECASE):
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
                try:
                    data.wien2k_version = version_file.read_text().strip().split()[0]
                except Exception:
                    pass

        # nbands fallback
        if data.nbands is None and data.nmat > 0:
            data.nbands = max(10, data.nmat // 2)

        return data


def parse_case_directory(path: Optional[Path] = None) -> CaseData:
    """Convenience function: parse all WIEN2k input files in a directory."""
    return CaseFileParser(path).parse_all()


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


def try_float(s: str) -> Optional[float]:
    """Try to parse a float, returning None on failure."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return None
