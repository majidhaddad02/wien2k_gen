"""
Electronic Structure Post-Processing Module for WIEN2k Output Files.

Parses WIEN2k output files (.output1, .energy, .scf, .klist_band,
.dos1ev, .dos2ev) to extract band structures, density of states (DOS),
and compute band gaps. Handles both spin-degenerate and spin-polarized
calculations with proper energy unit conversions.

All documentation and inline comments are in English per project standards.
"""

import os
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.constants import RYDBERG_TO_EV
from ..exceptions import ParsingError, MissingInputError


def _read_file_lines(filepath: Path) -> List[str]:
    """Read all lines from a file, stripping trailing whitespace."""
    if not filepath.exists():
        raise MissingInputError(f"Required WIEN2k output file not found: {filepath}")
    with open(filepath, "r") as fh:
        return [line.rstrip() for line in fh.readlines()]


def _extract_fermi_from_scf(case_name: str, path: str) -> float:
    """
    Extract the Fermi energy from the SCF output file.

    Parses ``case.scf`` (or ``case.scf2``) for the ":FER" line that
    contains the Fermi energy in Rydberg.

    Parameters
    ----------
    case_name : str
        WIEN2k case name (e.g., "TiC").
    path : str
        Directory containing the output files.

    Returns
    -------
    float
        Fermi energy in eV.
    """
    base = Path(path)
    scf_files = [
        base / f"{case_name}.scf2",
        base / f"{case_name}.scf",
    ]
    fermi_ry = 0.0
    found = False

    for scf_path in scf_files:
        if not scf_path.exists():
            continue
        lines = _read_file_lines(scf_path)
        for line in lines:
            if ":FER" in line and "F E R M I" not in line:
                tokens = line.strip(":").split()
                for i, tok in enumerate(tokens):
                    if tok in (":FER", "F E R M I"):
                        continue
                    try:
                        fermi_ry = float(tok)
                        found = True
                        break
                    except ValueError:
                        continue
                if found:
                    break
        if found:
            break

    if not found:
        fermi_ry = 0.0

    return fermi_ry * RYDBERG_TO_EV


def parse_band_structure(case_name: str, path: str) -> Dict[str, Any]:
    """
    Parse WIEN2k band structure from ``case.energy`` and ``case.klist_band``.

    Reads eigenvalues for every k-point and every band from the energy file,
    and k-point coordinates from the klist_band file. The Fermi level is
    read from the SCF output.

    Parameters
    ----------
    case_name : str
        WIEN2k case name.
    path : str
        Directory containing the output files.

    Returns
    -------
    dict
        Keys:
        - ``k_points`` (np.ndarray): k-point coordinates, shape (nkpt, 3).
        - ``k_labels`` (list[str]): labels from klist_band.
        - ``eigenvalues`` (np.ndarray): eigenvalues in eV, shape (nspin, nkpt, nbnd).
        - ``fermi`` (float): Fermi energy in eV.
        - ``nkpt`` (int): number of k-points.
        - ``nbnd`` (int): number of bands.
        - ``nspin`` (int): 1 for non-spin-polarized, 2 for spin-polarized.
    """
    base = Path(path)
    energy_path = base / f"{case_name}.energy"
    klist_path = base / f"{case_name}.klist_band"
    output1_path = base / f"{case_name}.output1"

    if not energy_path.exists():
        raise MissingInputError(f"Energy file not found: {energy_path}")

    energy_lines = _read_file_lines(energy_path)
    fermi_ev = _extract_fermi_from_scf(case_name, path)

    # Determine spin polarization by checking output1 for "SPIN"
    nspin = 1
    if output1_path.exists():
        output1_text = output1_path.read_text(errors="replace")
        if " SPIN" in output1_text:
            nspin = 2

    # Parse eigenvalues from case.energy
    eigenvalues_raw: List[List[float]] = []
    current_band: List[float] = []

    for line in energy_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if current_band:
                eigenvalues_raw.append(current_band)
                current_band = []
            continue
        if line.startswith("BAND"):
            if current_band:
                eigenvalues_raw.append(current_band)
                current_band = []
            continue

        # Data lines contain whitespace-separated floats
        tokens = line.split()
        for tok in tokens:
            try:
                val = float(tok)
                current_band.append(val)
            except ValueError:
                pass

    if current_band:
        eigenvalues_raw.append(current_band)

    # Spin-polarized files have alternating spin-up/spin-down blocks
    nkpt_est: int = 0
    nbnd_est: int = 0
    if eigenvalues_raw:
        longest = max(eigenvalues_raw, key=len)
        nbnd_est = len(longest) // 2 if eigenvalues_raw else 0

        # Count k-points from band blocks
        if nspin == 2:
            nkpt_est = len(eigenvalues_raw) // 2
        else:
            nkpt_est = len(eigenvalues_raw)

    # Reconstruct full matrix
    if nspin == 2:
        nkpt = nkpt_est
        # Determine nbnd from the max band length
        all_lengths = [len(b) for b in eigenvalues_raw]
        nbnd = max(all_lengths) if all_lengths else 0
        eigenvalues = np.zeros((nspin, nkpt, nbnd), dtype=np.float64)
        for ib in range(nbnd):
            for ik in range(nkpt):
                spin_up_idx = ib * 2 * nkpt + ik
                spin_dn_idx = ib * 2 * nkpt + nkpt + ik
                # This indexing is complex; use a simpler approach
                pass

        # Simpler: just parse all spin-up bands then all spin-dn bands
        half = len(eigenvalues_raw) // 2
        nbnd_up = max(len(b) for b in eigenvalues_raw[:half]) if half > 0 else 0
        nbnd_dn = max(len(b) for b in eigenvalues_raw[half:]) if half > 0 else 0
        nbnd = max(nbnd_up, nbnd_dn)
        eigenvalues = np.zeros((nspin, nkpt, nbnd), dtype=np.float64)

        # Fill spin-up
        for ib in range(min(nbnd_up, nbnd)):
            band_data = eigenvalues_raw[ib * 2] if ib * 2 < len(eigenvalues_raw) else []
            for ik in range(min(len(band_data), nkpt)):
                eigenvalues[0, ik, ib] = band_data[ik]
        # Fill spin-dn
        for ib in range(min(nbnd_dn, nbnd)):
            band_data = eigenvalues_raw[ib * 2 + 1] if ib * 2 + 1 < len(eigenvalues_raw) else []
            for ik in range(min(len(band_data), nkpt)):
                eigenvalues[1, ik, ib] = band_data[ik]
    else:
        nbnd = max(len(b) for b in eigenvalues_raw) if eigenvalues_raw else 0
        nkpt = len(eigenvalues_raw)
        eigenvalues = np.zeros((nspin, nkpt, nbnd), dtype=np.float64)
        for ib in range(nbnd):
            for ik in range(nkpt):
                if ik < len(eigenvalues_raw) and ib < len(eigenvalues_raw[ik]):
                    eigenvalues[0, ik, ib] = eigenvalues_raw[ik][ib]

    # Convert from Ry to eV
    eigenvalues *= RYDBERG_TO_EV

    # Parse k-points from klist_band
    k_points_list: List[List[float]] = []
    k_labels: List[str] = []

    if klist_path.exists():
        klist_lines = _read_file_lines(klist_path)
        for line in klist_lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) >= 3:
                try:
                    kx, ky, kz = float(tokens[0]), float(tokens[1]), float(tokens[2])
                    k_points_list.append([kx, ky, kz])
                    label = tokens[3] if len(tokens) > 3 else ""
                    k_labels.append(label)
                except ValueError:
                    continue

    k_points_arr = np.array(k_points_list, dtype=np.float64) if k_points_list else np.zeros((nkpt, 3))

    return {
        "k_points": k_points_arr,
        "k_labels": k_labels,
        "eigenvalues": eigenvalues,
        "fermi": fermi_ev,
        "nkpt": nkpt,
        "nbnd": nbnd,
        "nspin": nspin,
    }


def compute_band_gap(band_data: Dict[str, Any]) -> Tuple[float, bool, Optional[int], Optional[int]]:
    """
    Compute the band gap from pre-parsed band structure data.

    Determines the valence band maximum (VBM) and conduction band minimum (CBM)
    across all k-points, and classifies the gap as direct or indirect.

    Parameters
    ----------
    band_data : dict
        Output from :func:`parse_band_structure`.

    Returns
    -------
    tuple
        (gap_eV, is_direct, vbm_k_index, cbm_k_index)
        - ``gap_eV`` (float): band gap in eV (0.0 for metals).
        - ``is_direct`` (bool): True if gap is direct.
        - ``vbm_k`` (int or None): k-index of VBM.
        - ``cbm_k`` (int or None): k-index of CBM.
    """
    eigenvalues = band_data["eigenvalues"]
    fermi = band_data["fermi"]
    nspin = band_data["nspin"]
    nkpt = band_data["nkpt"]
    nbnd = band_data["nbnd"]

    if nspin == 2:
        # Use majority spin (up) for gap analysis
        energies = eigenvalues[0]
    else:
        energies = eigenvalues[0]

    # Find VBM: highest occupied band (energy <= fermi)
    vbm_per_k = np.full(nkpt, -np.inf)
    vbm_band_per_k = np.zeros(nkpt, dtype=int)
    for ik in range(nkpt):
        for ib in range(nbnd):
            e = energies[ik, ib]
            if e <= fermi + 0.01:  # small tolerance
                if e > vbm_per_k[ik]:
                    vbm_per_k[ik] = e
                    vbm_band_per_k[ik] = ib

    # Find CBM: lowest unoccupied band (energy > fermi)
    cbm_per_k = np.full(nkpt, np.inf)
    cbm_band_per_k = np.zeros(nkpt, dtype=int)
    for ik in range(nkpt):
        for ib in range(nbnd):
            e = energies[ik, ib]
            if e > fermi - 0.01:  # small tolerance
                if e < cbm_per_k[ik]:
                    cbm_per_k[ik] = e
                    cbm_band_per_k[ik] = ib

    # Global VBM and CBM
    vbm_energy = np.max(vbm_per_k)
    cbm_energy = np.min(cbm_per_k)

    gap_ev = max(0.0, cbm_energy - vbm_energy)

    vbm_k_idx: Optional[int] = int(np.argmax(vbm_per_k))
    cbm_k_idx: Optional[int] = int(np.argmin(cbm_per_k))

    is_direct = vbm_k_idx == cbm_k_idx

    return gap_ev, is_direct, vbm_k_idx, cbm_k_idx


def detect_semiconductor(gap: float) -> str:
    """
    Classify the material based on the band gap.

    Parameters
    ----------
    gap : float
        Band gap in eV.

    Returns
    -------
    str
        One of ``"metal"``, ``"semiconductor"``, or ``"insulator"``.
    """
    if gap < 0.05:
        return "metal"
    if gap < 3.0:
        return "semiconductor"
    return "insulator"


def parse_dos(case_name: str, path: str) -> Dict[str, Any]:
    """
    Parse WIEN2k density-of-states files.

    Reads ``case.dos1ev`` (or ``case.dos1evup`` / ``case.dos1evdn`` for
    spin-polarized) and ``case.dos2ev`` (and spin variants). Computes
    integrated DOS by cumulative trapezoidal integration from the lowest
    energy to the Fermi level.

    Parameters
    ----------
    case_name : str
        WIEN2k case name.
    path : str
        Directory containing the DOS output files.

    Returns
    -------
    dict
        Keys:
        - ``energies`` (np.ndarray): energy grid in eV.
        - ``dos_total`` (np.ndarray): total DOS (summed over spins if polarized).
        - ``dos_up`` (np.ndarray or None): spin-up DOS.
        - ``dos_dn`` (np.ndarray or None): spin-down DOS.
        - ``idos`` (np.ndarray): integrated DOS up to each energy point.
        - ``fermi`` (float): Fermi energy in eV.
        - ``nspin`` (int): 1 or 2.
    """
    base = Path(path)
    dos1_paths: List[Path] = [
        base / f"{case_name}.dos1evup",
        base / f"{case_name}.dos1evdn",
        base / f"{case_name}.dos1ev",
    ]

    nspin: int = 1
    dos1_up: Optional[Path] = None
    dos1_dn: Optional[Path] = None

    for fp in dos1_paths:
        if "up" in fp.name and fp.exists():
            dos1_up = fp
            nspin = 2
        elif "dn" in fp.name and fp.exists():
            dos1_dn = fp
            nspin = 2
        elif fp.exists():
            dos1_up = fp

    if dos1_up is None:
        raise MissingInputError(
            f"DOS file not found for case '{case_name}' in {path}. "
            f"Expected case.dos1ev or case.dos1evup."
        )

    def _parse_dos_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse a single DOS file returning (energies, dos_values)."""
        data = np.loadtxt(filepath, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        energies = data[:, 0] * RYDBERG_TO_EV
        # Column 3 (index 2) is total DOS, column 4 (index 3) is integrated DOS
        dos_values = data[:, 1] if data.shape[1] >= 2 else np.zeros_like(energies)
        return energies, dos_values

    energies, dos_up_arr = _parse_dos_file(dos1_up)

    dos_dn_arr: Optional[np.ndarray] = None
    if nspin == 2 and dos1_dn is not None:
        _, dos_dn_arr = _parse_dos_file(dos1_dn)

    if dos_dn_arr is not None and len(dos_dn_arr) != len(dos_up_arr):
        min_len = min(len(dos_up_arr), len(dos_dn_arr))
        dos_up_arr = dos_up_arr[:min_len]
        dos_dn_arr = dos_dn_arr[:min_len]
        energies = energies[:min_len]

    dos_total = dos_up_arr.copy()
    if dos_dn_arr is not None:
        dos_total += dos_dn_arr

    # Integrated DOS via trapezoidal rule
    idos = np.zeros(len(energies), dtype=np.float64)
    de = energies[1:] - energies[:-1]
    avg_dos = (dos_total[:-1] + dos_total[1:]) / 2.0
    idos[1:] = np.cumsum(avg_dos * de)

    fermi_ev = _extract_fermi_from_scf(case_name, path)

    return {
        "energies": energies,
        "dos_total": dos_total,
        "dos_up": dos_up_arr,
        "dos_dn": dos_dn_arr,
        "idos": idos,
        "fermi": fermi_ev,
        "nspin": nspin,
    }


def _parse_dos1_with_idos(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a dos1ev file that includes the integrated DOS column.

    Parameters
    ----------
    filepath : Path
        Path to the case.dos1ev file.

    Returns
    -------
    tuple
        (energies_eV, dos_values, idos_values).
    """
    data = np.loadtxt(filepath, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, data.size)
    energies = data[:, 0] * RYDBERG_TO_EV
    dos = data[:, 1] if data.shape[1] >= 2 else np.zeros_like(energies)
    idos_vals = data[:, 2] if data.shape[1] >= 3 else np.zeros_like(energies)
    return energies, dos, idos_vals


def _extract_gap_from_output1(case_name: str, path: str) -> Tuple[float, bool, Optional[int], Optional[int]]:
    """
    Attempt to read the reported band gap from case.output1.

    Parameters
    ----------
    case_name : str
        WIEN2k case name.
    path : str
        Directory path.

    Returns
    -------
    tuple
        (gap_eV, is_direct, vbm_k, cbm_k). Gap is 0.0 and k indices are None
        if not found.
    """
    base = Path(path)
    output1_path = base / f"{case_name}.output1"
    if not output1_path.exists():
        return 0.0, False, None, None

    lines = _read_file_lines(output1_path)
    gap_ev = 0.0
    is_direct = False
    vbm_k: Optional[int] = None
    cbm_k: Optional[int] = None

    gap_pattern = re.compile(
        r"^\s*:GAP\s*:\s*([\d\.\-]+)\s+Ry\s*=\s*([\d\.\-]+)\s+eV",
        re.IGNORECASE,
    )
    transition_pattern = re.compile(
        r"^\s*:TRANS\s*:\s*([\d]+)\s*->\s*([\d]+)",
        re.IGNORECASE,
    )
    direct_pattern = re.compile(r"direct", re.IGNORECASE)

    for line in lines:
        m = gap_pattern.match(line)
        if m:
            gap_ev = float(m.group(2))
        m2 = transition_pattern.match(line)
        if m2:
            if ":GAP" in line:
                vbm_k = int(m2.group(1))
                cbm_k = int(m2.group(2))
        if direct_pattern.search(line) and "gap" in line.lower():
            is_direct = True

    return gap_ev, is_direct, vbm_k, cbm_k


__all__ = [
    "parse_band_structure",
    "parse_dos",
    "compute_band_gap",
    "detect_semiconductor",
]
