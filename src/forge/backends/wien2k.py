"""
WIEN2k DFT backend — backward-compatible re-export shim.

All implementation lives in the forge.backends.wien2k package.
"""

from forge.backends.wien2k.core import Wien2kBackend, auto_detect_optimal_rkmax
from forge.backends.wien2k.parsers import DayfileResult, OutputParseResult

__all__ = [
    "DayfileResult",
    "OutputParseResult",
    "Wien2kBackend",
    "auto_detect_optimal_rkmax",
]
