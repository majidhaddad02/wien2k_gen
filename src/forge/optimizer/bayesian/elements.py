"""Periodic table data and chemical similarity for transfer learning."""


_ELEMENT_GROUPS = {
    1: {1},  2: {2},  3: {3, 11, 19, 37, 55},  4: {4, 12, 20, 38, 56},
    5: {5, 13, 31, 49, 81},  6: {6, 14, 32, 50, 82},
    7: {7, 15, 33, 51, 83},  8: {8, 16, 34, 52, 84},
    9: {9, 17, 35, 53, 85},  10: {10, 18, 36, 54, 86},
    11: {21, 39},  12: {22, 40, 72},  13: {23, 41, 73},
    14: {24, 42, 74},  15: {25, 43, 75},  16: {26, 44, 76},
    17: {27, 45, 77},  18: {28, 46, 78},  19: {29, 47, 79},  20: {30, 48, 80},
}
_ELEMENT_PERIODS = {
    1: {1, 2},  2: {3, 4, 5, 6, 7, 8, 9, 10},
    3: {11, 12, 13, 14, 15, 16, 17, 18},
    4: {19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
    5: {37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54},
    6: {55, 56, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86},
}

_ELEMENT_ELECTRONEGATIVITY = {
    1: 2.20, 2: 0.0,  3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44,
    9: 3.98, 10: 0.0, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58,
    17: 3.16, 18: 0.0, 19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66,
    25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01,
    33: 2.18, 34: 2.55, 35: 2.96, 36: 3.00, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33,
    41: 1.60, 42: 2.16, 43: 1.90, 44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69,
    49: 1.78, 50: 1.96, 51: 2.05, 52: 2.10, 53: 2.66, 54: 2.60, 55: 0.79, 56: 0.89,
    71: 1.27, 72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28,
    79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02, 84: 2.00, 85: 2.20, 86: 0.0,
}

_ELEMENT_COVALENT_RADIUS = {
    1: 32, 2: 28, 3: 128, 4: 96, 5: 84, 6: 76, 7: 71, 8: 66, 9: 57, 10: 58,
    11: 166, 12: 141, 13: 121, 14: 111, 15: 107, 16: 105, 17: 102, 18: 106,
    19: 203, 20: 176, 21: 170, 22: 160, 23: 153, 24: 139, 25: 139, 26: 132,
    27: 126, 28: 124, 29: 132, 30: 122, 31: 122, 32: 120, 33: 119, 34: 120,
    35: 120, 36: 116, 37: 220, 38: 195, 39: 190, 40: 175, 41: 164, 42: 154,
    43: 147, 44: 146, 45: 142, 46: 139, 47: 145, 48: 144, 49: 142, 50: 139,
    51: 139, 52: 138, 53: 139, 54: 140, 55: 244, 56: 215,
    71: 175, 72: 175, 73: 170, 74: 162, 75: 151, 76: 144, 77: 141, 78: 136,
    79: 136, 80: 132, 81: 145, 82: 146, 83: 148, 84: 140, 85: 150, 86: 150,
}

_GROUP_CRYSTAL_STRUCTURE = {
    1: "bcc",   2: "hcp",   3: "hcp",   4: "hcp",   5: "hcp",
    6: "bcc/fcc/hcp", 7: "diamond", 8: "fcc", 9: "fcc", 10: "fcc",
    11: "diamond", 12: "hcp", 13: "fcc", 14: "diamond", 15: "ortho",
    16: "ortho", 17: "ortho", 18: "fcc", 19: "bcc", 20: "fcc",
}

_ELEMENT_ATOMIC_NUMBERS = {
    "H": 1,  "He": 2,  "Li": 3,  "Be": 4,  "B": 5,   "C": 6,   "N": 7,   "O": 8,
    "F": 9,  "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,  "S": 16,
    "Cl": 17, "Ar": 18, "K": 19,  "Ca": 20, "Sc": 21, "Ti": 22, "V": 23,  "Cr": 24,
    "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32,
    "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39,  "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53,  "Xe": 54, "Cs": 55, "Ba": 56,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,  "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
}


def _get_element_group(atomic_number: int) -> int:
    """Map atomic number to simplified group number (1-20)."""
    for group_num, elements in _ELEMENT_GROUPS.items():
        if atomic_number in elements:
            return group_num
    return 0


def _get_element_period(atomic_number: int) -> int:
    """Map atomic number to period number (1-6)."""
    for period_num, elements in _ELEMENT_PERIODS.items():
        if atomic_number in elements:
            return period_num
    return 0


def _get_electronegativity_similarity(z1: int, z2: int) -> float:
    """Similarity based on Pauling electronegativity difference (0-1)."""
    en1 = _ELEMENT_ELECTRONEGATIVITY.get(z1, 1.5)
    en2 = _ELEMENT_ELECTRONEGATIVITY.get(z2, 1.5)
    if en1 == 0.0 or en2 == 0.0:
        return 0.5  # Unknown, assume moderate similarity
    diff = abs(en1 - en2)
    return max(0.0, 1.0 - diff / 3.5)


def _get_covalent_radius_similarity(z1: int, z2: int) -> float:
    """Similarity based on covalent radius ratio (0-1)."""
    r1 = _ELEMENT_COVALENT_RADIUS.get(z1, 150)
    r2 = _ELEMENT_COVALENT_RADIUS.get(z2, 150)
    ratio = min(r1, r2) / max(1, max(r1, r2))
    return max(0.0, ratio)


def _chemical_similarity(source_atomic_num: int, target_atomic_num: int) -> float:
    """
    Compute chemical similarity weight (0-1) between two elements.

    Multi-factor approach incorporating atomic properties:
    - Atomic number proximity (20% weight)
    - Same group (20% weight)
    - Same period (10% weight)
    - Electronegativity similarity (25% weight)
    - Covalent radius similarity (25% weight)

    These features are transferable between chemically similar systems
    for Bayesian optimization priors.

    Args:
        source_atomic_num: Atomic number of source element.
        target_atomic_num: Atomic number of target element.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if source_atomic_num == target_atomic_num:
        return 1.0

    max_z = max(source_atomic_num, target_atomic_num)
    z_distance = abs(source_atomic_num - target_atomic_num)
    z_similarity = max(0.0, 1.0 - z_distance / max(1.0, float(max_z)))

    same_group = 1.0 if _get_element_group(source_atomic_num) == _get_element_group(target_atomic_num) else 0.0
    same_period = 1.0 if _get_element_period(source_atomic_num) == _get_element_period(target_atomic_num) else 0.0
    en_sim = _get_electronegativity_similarity(source_atomic_num, target_atomic_num)
    r_sim = _get_covalent_radius_similarity(source_atomic_num, target_atomic_num)

    weight = 0.20 * z_similarity + 0.20 * same_group + 0.10 * same_period + \
             0.25 * en_sim + 0.25 * r_sim
    return min(1.0, max(0.0, weight))


__all__ = ["_chemical_similarity"]
