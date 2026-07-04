"""
Physical and computational constants used throughout the wien2k_gen package.
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

__all__ = [
    "RYDBERG_TO_EV",
    "HARTREE_TO_EV",
    "BOHR_TO_ANGSTROM",
    "KB_EV",
    "ELECTRON_MASS",
    "ELEMENTARY_CHARGE",
]
