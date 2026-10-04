"""
Physical and computational constants used throughout the forge package.
Centralizing these eliminates magic-number duplication across the codebase.
"""

# Rydberg to electron-volt conversion factor (CODATA 2018)
RYDBERG_TO_EV = 13.6056980659

# Hartree to eV
HARTREE_TO_EV = 27.2113961318

# Bohr radius in Angstroms
BOHR_TO_ANGSTROM = 0.529177210903

# Boltzmann constant in eV/K
KB_EV = 8.617333262145e-5

# Electron mass (SI units, kg)
ELECTRON_MASS = 9.1093837015e-31

# Elementary charge (C)
ELEMENTARY_CHARGE = 1.602176634e-19

# Reduced Planck constant (J·s)
HBAR = 1.054571817e-34

# HBAR^2 / ELECTRON_MASS in eV·Å² for effective mass calculation:
# m* / m_e = HBAR2_OVER_ME_EV_ANG2 / (d²E/dk²)
HBAR2_OVER_ME_EV_ANG2 = 7.619964

# Light hard elements (N, O, F). WIEN2k User Guide / Blaha et al. (2020)
# J. Chem. Phys. 152, 074101: small-RMT first-row p-block anions have hard
# potentials and need RKMAX >= 7.0. H is not in this class. S/Cl/Br are
# handled by the separate RMT-ratio heuristic in optimizer.parallel.
HARD_ELEMENTS = frozenset({7, 8, 9})

__all__ = [
    "BOHR_TO_ANGSTROM",
    "ELECTRON_MASS",
    "ELEMENTARY_CHARGE",
    "HARD_ELEMENTS",
    "HARTREE_TO_EV",
    "HBAR",
    "HBAR2_OVER_ME_EV_ANG2",
    "KB_EV",
    "RYDBERG_TO_EV",
]
